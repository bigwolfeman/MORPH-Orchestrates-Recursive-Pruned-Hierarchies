# tul_information

A Lean 4 development of the information bounds that apply to MORPH's slot loop.

It exists to settle one question with a proof instead of an argument: can a pass of a
deterministic loop add information about the target? It cannot. Everything the slot loop
earns must therefore be a relay (new inputs enter each pass) or extractability (the same
information re-arranged so the bounded reader can use it). The note that uses these
theorems is
[`.agents/notes/proposed/architecture/2026-09-13-information-view-of-the-slot-loop.md`](../../../.agents/notes/proposed/architecture/2026-09-13-information-view-of-the-slot-loop.md).

## Build

```sh
export PATH=$HOME/.elan/bin:$PATH
cd lab/theory/tul_information
nice -n 19 taskset -c 0,1 env LEAN_NUM_THREADS=1 lake build
nice -n 19 taskset -c 0,1 env LEAN_NUM_THREADS=1 lake env lean Axioms.lean
```

`lake build` exits 0 with zero warnings and zero errors. `Axioms.lean` prints one line per
theorem; every line must end `depends on axioms: [propext, Classical.choice, Quot.sound]`.
A `sorryAx` in any line means that theorem is unproved. There is no `sorry` in the source.

`taskset -c 0,1` holds the build to two cores. That is a house rule for this machine (the
UPS trips when the GPU is loaded and the CPU joins it), not a Lean requirement.

Mathlib `v4.31.0` comes from the local cache. `.lake/` is a symlink to
`/home/wolfe/lean-build/tul_information/.lake` because the repository lives on a slow
spinning disk. Delete the symlink and run `lake exe cache get` to rebuild it anywhere.

## The model

Finite alphabets. A joint distribution is a non-negative `p : α → β → ℝ` summing to one
(`TulInformation.Joint`). Mutual information is
`I(P) = ∑ a, ∑ b, p a b * log (p a b / (pL a * pR b))` in nats. Lean's `Real.log 0 = 0` and
`x / 0 = 0` give the `0 * log 0 = 0` convention without a side condition.

A channel here is always a deterministic map of the FIRST coordinate, `Joint.map`, with the
target `Y` untouched. That is the only channel the slot loop is: a pass rewrites the state
and does not touch the text.

## What is proved

| result | file:line | statement |
|---|---|---|
| `sum_mul_log_div_le_zero` | `TulInformation/Basic.lean:40` | Gibbs: `∑ p log (q/p) ≤ 0` for a pmf `p`, a sub-pmf `q`, `q` absolutely continuous w.r.t. `p` |
| `Joint.mi_nonneg` | `TulInformation/Basic.lean:309` | `0 ≤ I(P)` |
| `Joint.map_id` | `TulInformation/Basic.lean:276` | `P.map id = P` |
| `Joint.map_comp` | `TulInformation/Basic.lean:281` | `(P.map f).map g = P.map (g ∘ f)` |
| **`Joint.mi_map_le`** | `TulInformation/Basic.lean:233` | **the deterministic data-processing inequality: `I(f(Z);Y) ≤ I(Z;Y)`** |
| `Joint.condEntropy_map_ge` | `TulInformation/Basic.lean:363` | its log-loss form: `H(Y|Z) ≤ H(Y|f(Z))` |
| `Joint.mi_iterate_le` | `TulInformation/SlotLoop.lean:56` | `I(f^[n](Z);Y) ≤ I(Z;Y)` for every `n`: no number of passes adds information |
| `Joint.condEntropy_iterate_ge` | `TulInformation/SlotLoop.lean:64` | the same in log-loss form |
| **`Joint.mi_traj`** | `TulInformation/SlotLoop.lean:80` | **`I((Z, f Z, …, f^[n] Z); Y) = I(Z;Y)`: the trajectory carries exactly what the entry carries** |
| `Joint.mi_exit_le_traj` | `TulInformation/SlotLoop.lean:90` | the exit state carries no more than the trajectory |
| `Joint.mi_factor_le` | `TulInformation/SlotLoop.lean:102` | reach bound: if the read-out factors through a projection `π`, it carries at most `I(π(Z);Y)` |
| `Joint.mi_relay_le` | `TulInformation/SlotLoop.lean:115` | a pass carries at most what its inputs carry |
| **`Joint.mi_relay_redundant`** | `TulInformation/SlotLoop.lean:128` | **`I((F(Z,N), Z, N); Y) = I((Z,N); Y)`: a reader that already sees the pass's inputs gains exactly zero from the relayed state** |

The proof of the data-processing inequality is the finite one: the difference
`I(Z;Y) − I(f(Z);Y)` is a Kullback-Leibler divergence of the joint against a surrogate
`r a b = pL a * Q (f a) b / QL (f a)` that re-splits the image mass across each fibre in
proportion to the source marginal, and Gibbs closes it. `Joint.surrogate` and
`Joint.mi_term_diff` are that argument.

## What is assumed, and what is NOT proved here

* **Finite alphabets.** Real slot states are float vectors, hence a finite set, so this is
  not a restriction in principle. The theorems are qualitative: they bound nothing
  numerically.
* **`condEntropy` is DEFINED as `H(Y) − I(X;Y)`.** That it equals the Bayes log-loss of
  predicting `Y` from `X` is standard and is not proved here. The note relies on that
  identification when it talks about cross-entropy.
* **Independent randomness.** The Poisson depth draw and dropout are extra inputs. That an
  input independent of `Y` carries nothing about `Y` is standard and is not proved here.
  `mi_relay_le` covers the weaker statement that is proved: the pass carries at most what
  its inputs, randomness included, carry.
* **No continuity, no estimator theory.** Nothing here says how to estimate `I` from 480
  validation rows, and nothing here bounds what a TRAINED reader extracts. That gap is the
  whole content of "extractability" in the note, and it is measured, not proved.
