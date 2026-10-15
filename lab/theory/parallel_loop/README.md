# parallel_loop

A Lean 4 development of one question from Wolfe (2026-10-15): is there a parallel(ish) loop
topology, where the coda keeps a path to the prelude state around the loop, in which the main
next-token CE alone still makes the loop's extra passes pay?

It builds on [`../tul_pseudotoken/`](../tul_pseudotoken/) `Bypass.lean`
(`bypass_zeroes_passes`, `deeper_never_more_informative`): a positive K1-K6 is never
information, so the model here has a BOUNDED loop (a per-pass step limit) and a reader trained
with the loop (gradient flow). It does not import those projects; it needs only Mathlib.

The report that uses these theorems, with the predictions and the ranked implementations, is
`/home/wolfe/morph-scratch/abq/drawing-board/parallel-loop-lean.md` (private scratch).

## Build

```sh
export PATH=$HOME/.elan/bin:$PATH
cd lab/theory/parallel_loop
nice -n 19 taskset -c 15 env LEAN_NUM_THREADS=1 lake build
nice -n 19 taskset -c 15 env LEAN_NUM_THREADS=1 lake env lean Axioms.lean
```

Recorded 2026-10-15 after deleting `.lake/build`: `lake build` exits 0, "Build completed
successfully (1967 jobs)", zero warnings, zero errors. `lake env lean Axioms.lean` exits 0 and
prints 58 lines, all ending `depends on axioms: [propext, Classical.choice, Quot.sound]`.
`grep -rn "sorry\|admit\|^axiom" ParallelLoop/` finds nothing. No axiom is added.

Mathlib `v4.31.0`. `.lake/` is a symlink to `/home/wolfe/lean-build/parallel_loop/.lake`; its
`packages/` is a copy of `tul_information`'s (no Mathlib file was compiled).

## The two models

**PL1, the bounded-step depth law (`ParallelLoop/DepthLaw.lean`).** A pre-norm residual block
moves its stream by a bounded amount. So a loop pass is a step of length at most `δ` in a metric
space. The coda accepts a set `G` of good loop states and its loss at depth `T` is a monotone
function `φ` of the distance to `G`. The loop's WORK `W` is the entry state's distance to `G`.

**PL2, the linear-Gaussian gradient-flow model (`ParallelLoop/Prune.lean`).** The loop output
after `T` passes is `s_T β z` (Parcae's injection with a uniform decay builds a SCALE,
`s_T = (1 - a^T)/(1 - a)`). The reader reads it with weight `w` (loop product `c = w β`) and,
in the parallel topology, reads a skip carrying `z` with weight `v`. Squared loss averaged
over a finite depth law.

## What is proved

| result | file:line | statement |
| --- | --- | --- |
| `rem_le_infDist` | `DepthLaw.lean:101` | after `T` passes of step `≤ δ` the loop is still `≥ max(W - Tδ, 0)` from `G` |
| `greedy_infDist` | `DepthLaw.lean:151` | one weight-shared map attains that bound at every depth at once |
| `minimizer_forced` | `DepthLaw.lean:196` | so every minimizer of a depth-averaged loss has the forced per-depth loss |
| `kcurve_pos_iff` | `DepthLaw.lean:240` | the forced K-curve between depths `a < b` is positive iff `W > aδ` |
| `tokens_kcurve_pos_iff` | `DepthLaw.lean:265` | summed over tokens: positive iff some token's job is longer than `a` passes |
| `skip_work_eq_entry_work` | `DepthLaw.lean:297` | a coda reading `u + h` gives the loop the work of a series loop entered at `h` |
| `rem_mono`, `kcurve_mono_sq` | `DepthLaw.lean:71`, `:309` | more work: loss at least as high at every depth, K-curve at least as high |
| `rem_thin`, `thin_core_earns` | `DepthLaw.lean:319`, `:330` | a core `m` times thinner: equal loss at equal travel; earns where the full core is flat |
| `nrm_smul_pos`, `normalized_build_flat` | `DepthLaw.lean:348`, `:357` | a normalized read erases a build that only grows the scale |
| `par_minimizer` | `Prune.lean:134` | the parallel optimum puts zero weight on a depth-scaled duplicate of the skip |
| `hasDerivAt_Jpar_v/_w/_β` | `Prune.lean:172-203` | `gv`, `β gs`, `w gs` are minus half the partial derivatives |
| `par_stationary_prunes` | `Prune.lean:222` | every stationary point of the parallel flow has `w β = 0` |
| `flow_prune_slice`, `flow_prune_lyapunov` | `Prune.lean:232`, `:239` | once the skip fits, `ċ = -(w² + β²) var c` |
| `scaled_skip_grad_prunes` | `Prune.lean:262` | a scaled skip backward (gradient routing) has the same stationary set |
| `Jpar_fixed_eq_Jser` | `Prune.lean:281` | a frozen skip makes the loop a series loop on the remainder |
| `gs_zero_par`, `gs_zero_ser` | `Prune.lean:254`, `:286` | at `c = 0` the loop gradient is `S1 gv` (parallel) vs `κ S1` (series) |
| `ser_minimizer`, `ser_kcurve_pos` | `Prune.lean:297`, `:314` | the series optimum keeps the loop and has a positive K-curve |
| `par_le_ser_every_depth` | `Prune.lean:325` | ...on a loss never below the parallel optimum's (zero) at any depth |
| `par2_minimizer`, `unique_share` | `Prune.lean:346`, `:373` | parallel keeps the loop-only channel; its K-curve is the unique share `η²/(κ²+η²)` of series' |
| `drop_minimizer`, `drop_eval_loss`, `drop_tax` | `Prune.lean:400`, `:433`, `:440` | a stochastic skip keeps `c_p` but pays `c_p² var > 0` at eval, every depth but the mean |

What is MODELLED and not proved: that a Parcae core pass is a bounded step in a metric where the
coda's loss is monotone in the distance to a good set; that the skip arm's RMSNorm sees a
uniform-decay build (exact only with a fixed block output); the size of the loop's unique job
on OpenWebText at 5k. The PL2 dynamics results are algebraic (stationary sets, the sign of a
time derivative on a slice); no theorem here says gradient flow converges.

## Numerics

`numerics/toy.py` (numpy, one thread, about 3 min) trains both models by SGD/Adam and prints
`numerics/toy_output.txt`. Part 1 matches the PL2 closed forms (e.g. unique share 0.2051 vs
0.2000; stochastic-skip tax 0.00526 vs 0.00527). Part 2 trains a nonlinear loop with a hard step
bound: series earns (K1-K6 +5.4), parallel is near flat when the loop-only job fits one pass
(+0.024), the thin core earns more (+0.053) and matches the full core at equal travel, and the
withheld and stochastic skips raise K1-K6 only with a loss at least as high at every depth.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 taskset -c 15 \
  /home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python numerics/toy.py
```
