# Cycling spiral: the hierarchy repeated per loop, with Parcae or HRM credit

Status: planned (written 2026-10-16, before any GPU run of the arms)

## Question

Does a loop that repeats the whole coarse-to-fine hierarchy use depth where a single descent does not?
And is the gain, if any, from the moving hierarchy or from deep supervision?

## Hypothesis

I think a loop under fixed conditioning converges in a pass or two, which is why the spiral's sampled level
reads only +0.007 and a span level reads flat. Repeating the whole hierarchy moves the fixed point: each
cycle's coarse levels pool a state that holds the fine levels' last inferences, so the next cycle has new work
(a multigrid V-cycle; HRM's two timescales). Sampling the start level and skipping levels (Wolfe) keeps each
level able to act from any state. Deep supervision (a CE after every cycle, state detached between cycles) is
the HRM credit rule; an analysis of HRM credits its results to that rule, not to the H/L split, so a plain
Parcae loop with deep supervision is the arm that separates the two.

## Arms (branch lxtul/cycle; levels 16, 8, 4, 2; cycle count Poisson mean 3, max 6; 5k steps, seeds 1 and 2)

| arm | credit | start level | skip p |
| --- | --- | --- | --- |
| cycle, end credit | CE after the last cycle (Parcae) | uniform | 0.2 |
| cycle, deep credit | CE after every cycle, detached carry (HRM) | uniform | 0.2 |
| cycle, fixed schedule | end | coarsest | 0 |
| cycle, deep, fixed schedule | deep | coarsest | 0 |
| plain Parcae, deep supervision | CE after every core pass, detached carry | (token loop) | (none) |

Controls (already run, same paired rows): published spiral, spiral + Poisson at chunks of 2, plain Parcae + pointer.

## Predictions

1. Main: the cycling arms' K-curve over cycles slopes down more steeply than spiral + Poisson-2's curve over its
   passes, with CE the same as spiral + Poisson-2 or better (inside its seed spread counts as the same).
2. Deep credit against end credit, and plain + deep supervision against plain Parcae: I predict the direction only
   for the cycle arms (deep slopes down more steeply than end, because every cycle is graded). For plain + deep
   supervision I cannot say which way CE moves; I report it beside plain Parcae.
3. Schedule sampling (uniform start, skips) gives the same CE as the fixed schedule or better.

If 1 is supported: the moving fixed point is the lever, and the next arm puts a span level inside the cycle and
then a gate over the schedule. If 1 is not supported: repetition of the hierarchy is not what the spiral lacks,
and the XM span plan (a non-optional span job) is the next test instead.

Gates (not predictions): finite losses, peak memory under 26 GiB in every smoke, paired scored tokens 510003,
head-off reads worse than head-on.

## Method

Code and CPU tests from a sonnet builder (`spiral.cycle`, `plain.deep_supervision`; test_spiral_cycle.py 31 passed,
test_spiral.py 164 passed, test_bench_all.py 35 passed, a scratch HEAD comparison 17 passed). GPU steps:
`morph-scratch/abq/cycle/gpu_todo.sh` (pytest, smokes, bench, 5k runs, head-off, paired eval on the minfan-5k rows).

## Amendment 1 (2026-10-16, before any 5k run)

All five smokes ran out of memory (end credit peaked at 28.18 GiB; the rest OOMed at about 28.5 GiB). Two method
changes, both before any 5k run: (1) the end-credit arms truncate backprop to the last 3 cycles (`cycle.bptt: 3`,
Parcae's own n no-grad + k grad law; mean 3 and max 6 kept). (2) Deep credit runs one backward per cycle (HRM's
recipe), with the prelude output as a leaf and its summed gradient backpropagated at the end; an fp64 CPU test shows
the gradient equals the single-backward gradient on every parameter. The deep head uses the chunked CE. A 3-step
memory probe per arm now runs before each smoke. Predictions are unchanged.
