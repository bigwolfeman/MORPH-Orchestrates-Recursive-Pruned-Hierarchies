# lctul_euler_depth

A Lean 4 development of why the LCTUL Euler k-curve is flat.

LCTUL (Latent Coded TUL, `docs/tul-code-spec.md`) samples a slot's code with `k` Euler
steps of a flow-matching velocity field and calls `k` the loop depth. Every arm reads the
same k-curve: one step from `k = 1` to `k = 2` worth 0.05 to 0.11 nats, then at most 0.02
nats from `k = 2` to `k = 16`. This project proves, with Mathlib, that a flow-matching
Euler sampler on a Gaussian-like conditional code distribution produces exactly that curve,
and says what a code distribution must look like for depth to pay. The note that uses these
theorems is
[`.agents/notes/proposed/architecture/2026-09-16-lctul-euler-depth-theorem.md`](../../../.agents/notes/proposed/architecture/2026-09-16-lctul-euler-depth-theorem.md).

## Build

```sh
export PATH=$HOME/.elan/bin:$PATH
cd lab/theory/lctul_euler_depth
nice -n 19 taskset -c 0,1 env LEAN_NUM_THREADS=1 lake build
nice -n 19 taskset -c 0,1 env LEAN_NUM_THREADS=1 lake env lean Axioms.lean
```

Recorded 2026-09-16: `lake build` exits 0, "Build completed successfully (1993 jobs)",
zero warnings and zero errors in this project's files. `lake env lean Axioms.lean` exits 0
and prints 32 lines; every line ends `depends on axioms: [propext, Classical.choice,
Quot.sound]`. `grep -rn sorry LctulEulerDepth/` finds nothing.

`taskset -c 0,1` holds the build to two cores. That is a house rule for this machine (the
UPS trips when the GPU is loaded and the CPU joins it), not a Lean requirement.

Mathlib `v4.31.0` comes from the local cache. `.lake/` is a symlink to
`/home/wolfe/lean-build/lctul_euler_depth/.lake` because the repository lives on a slow
spinning disk. The package tree was copied from `tul_information`'s build directory; delete
the symlink and run `lake exe cache get` to rebuild it anywhere.
`tul_information` (`../tul_information`) is a path dependency: `Information.lean` reuses its
`Joint`, `mi`, `map` and the data-processing inequality `mi_map_le`.

## The model

`v z t` is the thinker's velocity field at ONE fixed context (tape plus seed). The context
is never varied inside a theorem, so it is not written. `eulerEndpoint v k z0` is
`morph/model/tul_code.py::euler_sample` (`z ← z + (1/k) v(z, j/k)` for `j = 0 … k-1`)
in a real normed space `E`.

Probability is finite, as in `tul_information`: a law on a finite index type is a weight
function summing to one (float vectors are a finite set). `F` is a real inner product space
(`EuclideanSpace ℝ (Fin 1024)` for one cell). The source `z0` is a weight `r` on a source
alphabet, the code `z1` a weight `q` on a code alphabet, and the CFM pair
`(z0, z1) ~ r ⊗ q` is independence, which is how `cfm_pair` draws `z0` (`torch.randn`, never
a function of the code).

## What is proved

| result | file:line | statement |
|---|---|---|
| `eulerEndpoint_one` | `LctulEulerDepth/Euler.lean:90` | one step from `t = 0` returns `z0 + v z0 0` |
| **`eulerEndpoint_one_of_meanField`** | `Euler.lean:97` | **if `v z 0 = m - z`, one step returns `m` from every source draw: the `k = 1` sample is a constant** |
| `euler_dirac` | `Euler.lean:108` | along the Dirac field `(z* - z)/(1 - t)` the Euler iterates are the exact straight line |
| **`eulerEndpoint_dirac`** | `Euler.lean:136` | **a code determined by the context is sampled exactly at every `k ≥ 1`** |
| `euler_affine`, `eulerEndpoint_affine` | `Euler.lean:158`, `:172` | an affine field `a t • z + b t` gives `z_k = λ_k • z0 + c_k`, `λ_k`, `c_k` independent of `z0` |
| `euler_gauss` | `Euler.lean:200` | along the isotropic-Gaussian field the shift after `j` steps is exactly `(j/k) • μ` |
| **`eulerEndpoint_gauss`** | `Euler.lean:233` | **on the isotropic-Gaussian field the `k`-step endpoint is `μ + λ_k • z0` for every `k ≥ 1`: the mean is exact at every depth, the noise direction is the source direction, and `k` sets one scalar** |
| `affScale_gauss_one`, `eulerEndpoint_gauss_one` | `Euler.lean:240`, `:245` | `λ_1 = 0`: the one-step sample is the bare mean |
| `affScale_gauss_two` | `Euler.lean:252` | `λ_2 = 1/2` on matched scales: the two-step sample is `μ + z0/2` |
| `affScale_gauss_pos` | `Euler.lean:282` | `λ_k > 0` for `k ≥ 2` on matched scales: from two steps on, the sample is never the mean |
| **`euler_error_bound`** | `Euler.lean:297` | **discrete Grönwall: if one Euler step from the exact path misses it by at most `δ` and the field is `L`-Lipschitz, the `k`-step iterate is within `δ ∑_{i<k} (1 + L/k)^i` of the exact one** |
| `euler_exact_of_zero_defect` | `Euler.lean:348` | `δ = 0` (the field is constant along the path) makes Euler exact at every `k` |
| `euler_error_bound_curvature` | `Euler.lean:374` | with `δ = M/(2k²)` the endpoint error is at most `M e^L / (2k)`: depth pays at the rate the field curves along the path |
| `sum_norm_sub_sq`, `sum_norm_sub_sq_ge` | `SecondMoment.lean:62`, `:83` | the variance decomposition `∑ wᵢ ‖c - yᵢ‖² = ‖c - m‖² + Var`; the mean is the unique minimiser |
| **`fm_loss_t0`** | `SecondMoment.lean:101` | **the CFM loss at `t = 0` with an independent source is `∑ₐ rₐ ‖(xₐ + g xₐ) - m‖² + Var(z1)`: it charges the field exactly for the distance between the one-step endpoint and the code's mean** |
| `fm_loss_t0_ge`, `fm_loss_t0_optimal`, `fm_loss_t0_optimal_unique` | `SecondMoment.lean:116`, `:125`, `:133` | the floor is the code's variance; `g x = m - x` reaches it; only `x + g x = m` does |
| **`k1_endpoint_of_fm_optimal`** | `SecondMoment.lean:151` | **with the optimal field, the `k = 1` Euler endpoint from ANY source point is the conditional mean** |
| `residual_two_draws` | `SecondMoment.lean:160` | two independent draws from one law sit at `2 × Var` from each other |
| `total_variance` | `SecondMoment.lean:175` | total = within + between, across contexts |
| **`residual_ratio_conditional_draw`** | `SecondMoment.lean:197` | **a draw from the conditional law reads residual `2 W`, the total is `W + B`: the ratio is `2 (1 - R²)`; the conditional mean reads `1 - R²`** |
| `explained_of_residual_ratio` | `SecondMoment.lean:213` | residual ratio `ρ` means the context explains `1 - ρ/2` of the code: `1.8` is one tenth |
| `norm_wmean_le`, `norm_wmean_le_of_shell` | `SecondMoment.lean:223`, `:234` | the mean of shell points is inside the shell: the `k = 1` code is off the RMS shell the coda trained on |
| `Joint.prodIndep`, `Joint.mi_prodIndep` | `Information.lean:39`, `:66` | independent noise carries nothing: `I((ctx, z0); span) = I(ctx; span)` |
| **`Joint.sample_mi_le`**, `sample_mi_le_depth` | `Information.lean:88`, `:95` | **`I(F(ctx, z0); span) ≤ I(ctx; span)` for every sampler `F`, hence for every Euler depth** |
| `Joint.sample_condEntropy_ge` | `Information.lean:107` | the log-loss form: no depth lowers the Bayes conditional entropy below `H(span | ctx)` |

## What the theorems say about the design

1. **`k = 1` is the conditional mean, not a sample** (`fm_loss_t0_optimal_unique`,
   `k1_endpoint_of_fm_optimal`). The loss at `t = 0` is minimised only by `v(z0, 0) = m - z0`,
   and one Euler step from `t = 0` then lands on `m` whatever `z0` was. Measured: at `k = 1`
   the 8-draw marginal equals the single-draw mean (4.971 against 4.973 at 5k); the draws do
   not differ because the endpoint does not depend on the draw. The mean sits inside the RMS
   shell (`norm_wmean_le_of_shell`) and after `code_rmsnorm` has no noise component, so it is
   a different object from every code the coda was trained on. That is the `k = 1 → 2` step:
   mean against sample, once.
2. **On a Gaussian-like conditional, depth past `k = 2` moves one scalar**
   (`eulerEndpoint_gauss`). The endpoint is `μ + λ_k • z0` at every `k`. The mean component is
   exact at `k = 1` already; `k` only grows the radius `λ_k` from `0` through `1/2` toward the
   code's own spread. Nothing about the span enters through `k`. The simulation
   (`sim/kcurve.py`, below) shows the same on the exact conditional-expectation field: the
   residual against an independent true code climbs `1.13 → 1.68` from `k = 2` to `16` at
   `R² = 0.1`, the cosine to the mean falls `0.54 → 0.33`, and the exact flow reads `1.81`.
3. **A determined code is sampled in one step** (`eulerEndpoint_dirac`). The other limit
   gives depth nothing either.
4. **Depth pays exactly as much as the field curves along the path** (`euler_error_bound`,
   `euler_error_bound_curvature`). For the L²-optimal field that curvature exists only where
   the conditional law is not Gaussian: separated modes make the posterior weights turn the
   field. The simulation shows it: with modes separated by about seven within-mode
   standard deviations the endpoint's log-density under the true law climbs by 8 to 27
   nats from `k = 2` to `16` and the mode-hit rate goes `0.45–0.85 → 1.00`; with the
   separation at two standard deviations the gain is 3 nats to `k = 4` and then reverses;
   at spread `0.5` against std `0.3` the k-curve is flat (`−0.4`). The count of modes is not
   the obstacle (4096 separated modes in 16 dimensions still pay); the separation against
   the within-mode noise is.
5. **The residual reading is the second-moment prediction** (`residual_ratio_conditional_draw`).
   A sampler that lands ON the conditional law reads `2 (1 - R²)` on
   `code_subspace_probe.py`; `1.7–1.9` means the context explains 5 to 15 % of the code's
   variance. The exact flow in the simulation reads `1.81` at `R² = 0.1` and `1.00` at
   `R² = 0.5`. The thinker's sample is where a correct conditional sampler of a nearly
   unexplained code would be; it is not a training failure the probe can see.
6. **No depth adds information** (`sample_mi_le`). The sampled cell at any `k` is
   `F(tape, seed, z0)` with `z0` independent of everything, so it carries at most
   `I(tape, seed; span)`. The strict coda's floor is `H(span | tape, seed)`, which the
   cross-span budget bounds at the ruler minus 0.40 nats per token, whatever `k` is.

What follows for LCTUL. The code is a near-verbatim copy of a span whose entropy given the
past is about 4.4 nats per token over about 12 tokens; the past removes at most 0.40 nats
per token of it. The conditional `p(z | tape, seed)` is therefore a mixture of about `e^50`
near-orthogonal unit-RMS codes of nearly equal weight, blurred by the noise augmentation at
`0.5–1.0` on a unit scale. Its L²-optimal field is, to the accuracy any thinker reaches on
it, the affine mean field of theorem 2, and the measured residual says the trained field is
one. Under theorem 2 the k-curve of that field is one scalar, and under theorem 1 the
`k = 1 → 2` step is the only event. The flat k-curve is the design's prediction.

The queued LaDiR chain raises the noise to `3.0`, which lowers the separation of the
code's modes against their blur from about two within-mode standard deviations to about
one third of one. Under theorems 2 and 4 that can only flatten the k-curve further; it can
change what the frozen coda tolerates (sample quality), not the shape of the curve past
`k = 2`. LaDiR's latent is a code of a reasoning step whose conditional given the problem
and the earlier steps has few, separated modes, and its score is a discrete accuracy for
which landing in a mode against between modes is the whole game. That is the property
LCTUL's latent lacks: the object it codes has high entropy given the past. A code that can
earn Euler depth must be a code of something the past nearly determines, with a few
separated outcomes, and a coda whose tolerance radius is smaller than the separation.

## The simulation

`sim/kcurve.py` (`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 19 taskset -c 0,1 python
sim/kcurve.py`, 55 s on two cores, numpy 2.5.2, exit 0; output in `sim/kcurve_output.txt`)
runs `euler_sample` on the EXACT conditional-expectation field of (a) a Gaussian conditional
and (b) separated-mixture conditionals, so the k-curve it reads is the design's, with no
optimisation gap. Columns: `logp` the endpoint's log-density under the true conditional,
`resid` the `code_subspace_probe` ratio against an independent true draw, `radius` the
`λ_k` of `eulerEndpoint_gauss`, `cos_mu` the cosine to the conditional mean, `hit` the
fraction of endpoints inside a mode.

(a) Gaussian, `R² = 0.1`: radius `0.000, 0.468, 0.674, 0.797, 0.864, 0.918` at
`k = 1, 2, 4, 8, 16, 64`, exact flow `0.937`; resid `0.905 → 1.812`; cos_mu `1.000 → 0.302`.
(b) mixture, 8 modes, spread `2.0`, std `0.3`: logp `−138, −12.5, −3.2, −3.5, −4.3, −5.1`
(true-code reference `−5.55`), hit `0.00, 0.85, 1.00, 1.00, 1.00, 1.00`. The same at std
`1.0`: logp `−41, −27, −24.1, −23.8, −24.1, −24.6`, the gain past `k = 4` gone.

## What is assumed, and what is NOT proved here

* **The Gaussian field is taken as given.** `gaussCoeff` is the coefficient of the Gaussian
  conditional expectation `E[z1 - z0 | z_t]`; that the L²-optimal field for a Gaussian pair is
  this affine map is standard Gaussian conditioning and is NOT proved. Everything about it
  in `Euler.lean` is a consequence of the affine form.
* **Finite weights, not Gaussian measures.** The second-moment identities are proved for
  weight functions on finite index types. The measure-theoretic versions over a Gaussian
  source are not proved; the finite version is what the float grid is.
* **The exact-flow limit `λ_k → s1/s0` is not proved.** The simulation shows it
  (`0.918` at `k = 64` against `0.937`); the Lean development stops at the closed form of
  `λ_k` and its positivity.
* **Nothing here says what a TRAINED field is.** That the thinker's field is near-affine is
  the measured residual (`1.7–1.9`), read through `residual_ratio_conditional_draw`; the
  theorems say what follows from it. A field that resolved `e^50` separated modes would
  curve, and theorem 4 says depth would then pay; the theorems do not say whether such a
  field is learnable by MSE regression on 2048-dimensional targets, and every arm says it
  was not.
* **The information bound is qualitative.** `sample_mi_le` bounds nothing numerically; the
  0.40-nat budget is a measurement (`lab/experiments/failures/2026-09-11-arc-span-budget.md`).
* **`condEntropy` is `H(Y) - I(X;Y)` by definition**, inherited from `tul_information`.
