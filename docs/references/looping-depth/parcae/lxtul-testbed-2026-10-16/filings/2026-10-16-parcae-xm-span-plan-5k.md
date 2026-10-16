# XM span plan: best-of-K plans from a span loop on the published spiral

Status: success (filed 2026-10-16; main prediction supported, effect small)

## Question

Does best-of-K selection of a span plan give the span loop a job, so that it uses depth on a valid bound?

## Hypothesis

I think the spiral's span level is depth-flat because its job is optional: it carries a past summary the coda
rebuilds from tokens. A plan picked best-of-K (Explorative Modeling, arXiv 2607.27372) carries future information
about the next span that the coda cannot get elsewhere, and only the loop's K candidates supply it. So the loop
has a reason to go deep, in proportion to how well it predicts the next thought. Design:
`morph-scratch/abq/drawing-board/xm-span-plan.md`. Wolfe expects it to be useless; it is the cheap test.

## Arms (branch lxtul/xm-plan; base published spiral; span loop at Poisson depth mean 6, cap 8; 5k, seeds 1, 2)

- XM-8 (`spiral_xm_plan8`): 8 plan heads, a parameter-free bag score through the tied head picks the winner per
  span, Sinkhorn-balanced in training, argmax at eval; fixed uniform prior over the index.
- Control (`spiral_xm_plan1`): the same with one head (ordinary training, no search).

Readouts: B(d) = CE given the winners + ln K per plan-conditioned span (a valid NLL bound); U(d) = CE with a
uniform random index path; plan-use gap = U - CE given the winners; SIS(6) (sequential importance sampling,
eval only); K1-K6 of B.

## Predictions

1. Main: the K-curve of B slopes down more steeply for XM-8 than for the control.
2. XM-8's plan-use gap is above 0 (the control's is 0 by construction).
3. On the bound itself, XM-8 is worse than the control at 5k, because the uniform prior charges ln 8 per span
   in full.

If 1 is supported: a K sweep, then the learned prior over the index (phase 2), then matched-FLOP shrinking of
the token path. If 1 is not supported and the gap is above 0: the score does not match what the coda uses; one
exact-matching probe, then close. If the gap is 0: the coda ignores the plan; close.

Gates (not predictions): finite losses; causality test; Sinkhorn balance; every index wins within each 500 steps;
scored token counts match; SIS at K = 1 equals CE; peak under 26 GiB. Speed is reported, not graded.

## Method

Code and CPU tests from a sonnet builder; GPU steps from `morph-scratch/abq/xm/gpu_todo.sh`, which waits for the
cycling chain to finish.

## Results

Code 8e6959f on lxtul/xm-plan. Chain `morph-scratch/abq/xm/gpu_todo.sh`, log `gpu_todo.log`. All gates held: finite
losses, peak 16.4 GiB, scored tokens 510003 OK, head-off worse than head-on on every run, SIS at K = 1 equals CE
(3.70406 vs 3.70405). Paired eval: `morph-scratch/abq/xm/eval/paired_xm.json` (480 minfan-5k rows). SIS:
`eval/*-sis.json`. Bench: `eval/bench_xm.md`.

| run | CE@6 given the chosen plan | B@6 (bound) | SIS(6) | U@6 (uniform plan) | K1-K6 of B | plan-use gap @6 |
|---|---|---|---|---|---|---|
| XM-8 seed 1 | 3.6855 | 3.7890 | 3.7292 | 3.8514 | +0.0009 | 0.166 |
| XM-8 seed 2 | 3.6819 | 3.7854 | 3.7211 | 3.8115 | +0.0014 | 0.130 |
| control seed 1 | 3.7041 | 3.7041 | 3.7041 | 3.7041 | +0.0002 | 0 |
| control seed 2 | 3.7033 | 3.7033 | 3.7033 | 3.7033 | +0.0001 | 0 |
| published spiral (2 seeds, same rows) | 3.6905 / 3.6919 | | | | 0 | |
| spiral + Poisson-2 (2 seeds) | 3.6880 / 3.6933 | | | | +0.0082 / +0.0059 | |
| plain Parcae + pointer (1 seed) | 3.7256 | | | | +0.0377 | |

Speed (bench, batch 6): XM-8 trains at 190.2 ms per step and 11.00 TFLOP; the control 164.6 ms and 10.51 TFLOP. The
published spiral is 138.0 ms and 8.73 TFLOP (`morph-scratch/abq/spiral/eval/bench_v2.md`). Inference 46.9 vs 38.7 ms.

1. Main (B's K-curve steeper at K = 8 than at K = 1): held. XM-8 reads +0.0009 and +0.0014; the control reads +0.0002
   and +0.0001. The seed ranges do not overlap. The effect is small: about one sixth of spiral + Poisson-2's K-curve
   and one thirtieth of plain Parcae's.
2. Plan-use gap above 0: held (0.166, 0.130). The coda depends on the plan: a uniform random plan path costs 0.13 to
   0.17 nats per token against the chosen one.
3. XM-8 worse than the control on the bound at 5k: held. B is about 0.083 worse. The tighter SIS estimate is still
   0.018 to 0.025 worse than the control.

Not predicted: the control itself reads 3.7037 (2-seed mean), 0.0125 worse than the published spiral on the same rows
(3.6912). One plan head and a span loop added to the spiral cost CE.

## Verdict

Supported, effect small. Best-of-8 selection makes the span loop's K-curve steeper than one head (+0.0012 vs +0.0002,
2-seed means), but the depth it earns is the smallest of any positive loop arm on this testbed, and the likelihood is
worse (SIS 3.725 vs 3.704).

## Updated hypothesis

A plan picked best-of-K does carry information the coda uses (gap 0.13 to 0.17). But the loop does not need depth to
make it: almost all of the plan's value is there at one pass. The selection gives the span loop a job, and the job is
shallow. The ln K cost is larger than what the plan saves at 5k. The planned next step for a supported result (a K
sweep, then a learned prior) is held for Wolfe's call, because the effect that would justify it is 0.001 nats.

## What this changes

Nothing in the shipped design. It is the cheap test Wolfe expected to be useless: the plan is used, the loop's depth
is not. A non-optional span job by itself does not make a span loop go deep. This agrees with the cycling question
still in flight: if depth comes from anywhere here, it is from the fixed point moving between passes, not from the
span job.

## Artifacts

`/home/wolfe/morph-scratch/abq/xm/eval/` (paired_xm.json, four SIS json, four head-off json, bench_xm.md),
`/home/wolfe/morph-scratch/abq/xm/gpu_todo.log`, runs `/home/wolfe/parcae-runs/spiral-xm-plan{8,1}-5k-s{1,2}`.
