# tul_exploration

A Lean 4 development of when a looped latent can contribute: when sampled width survives
training and helps, and when loop depth is necessary rather than merely possible.

It sits beside [`../tul_information/`](../tul_information/), which proved that a
deterministic pass cannot add information, so a loop earns only by a relay or by
extractability. This project asks the next two questions. Can WIDTH (several latents per
slot, sampled or enumerated) earn, and why did the learned noise of LXTUL-GK die? And what
must a task or an objective look like for DEPTH to be necessary when nothing restricts what
the reader sees? The note that uses these theorems is
[`.agents/notes/proposed/architecture/2026-09-23-provable-loop-contribution.md`](../../../.agents/notes/proposed/architecture/2026-09-23-provable-loop-contribution.md).

## Build

```sh
export PATH=$HOME/.elan/bin:$PATH
cd lab/theory/tul_exploration
nice -n 19 taskset -c 0,1 env LEAN_NUM_THREADS=1 lake build
nice -n 19 taskset -c 0,1 env LEAN_NUM_THREADS=1 lake env lean Axioms.lean
```

Recorded 2026-09-23 after a clean rebuild of this package: `lake build` exits 0 with
"Build completed successfully (2195 jobs)" and zero warnings. `lake env lean Axioms.lean`
exits 0 and prints 95 lines. 80 end `depends on axioms: [propext, Classical.choice,
Quot.sound]`. 15 (the `Prog` lemmas and most of `Hops.lean`) use a subset of those three or
none. No line mentions `sorryAx`, and `grep -rn sorry TulExploration/` finds nothing.

Recorded 2026-10-18 after the particle files (`TulExploration/Particles/`) were added,
built with `nice -n 19 taskset -c 15 env LEAN_NUM_THREADS=1 lake build` (one core):
"Build completed successfully (2204 jobs)", zero warnings. `lake env lean Axioms.lean`
exits 0 and prints 185 lines: 169 end `[propext, Classical.choice, Quot.sound]`, 16 use a
subset or none. No line mentions `sorryAx`.

`taskset -c 0,1` holds the build to two cores. That is a house rule for this machine (the
UPS trips when the GPU is loaded and the CPU joins it), not a Lean requirement.

Mathlib `v4.31.0` comes from the local cache. `.lake/` is a symlink to
`/home/wolfe/lean-build/tul_exploration/.lake` because the repository lives on a slow
spinning disk; its package tree was copied from `tul_information`'s. `tul_information` is a
path dependency (`Bayes.lean` reuses its Gibbs inequality, `sum_mul_log_div_le_zero`), so
`../tul_information/.lake` must exist too. In the main checkout it is already a symlink. In a
fresh worktree, point it at an empty directory on `/home` before the first build, or lake
writes that build onto the HDD. Delete the symlinks and run `lake exe cache get` to rebuild
anywhere.

## The model

Everything is finite, as in `tul_information`: a law is a weight function on a finite type
that sums to one. Float vectors are a finite set, so this is not a restriction in principle.

* **Reader.** A family of pmfs `p(y | z)` over the target, one per latent value, strictly
  positive (a softmax). "Hedging" means the family can output the true conditional law. The
  teacher-forced AR coda, which sees the true prefix and outputs a full categorical, is the
  hedging reader. A "committed" reader cannot hedge: a product over span positions (a
  parallel span decode) cannot represent the correlation between positions.
* **Width.** A draw hands the reader `K` latents. The smooth multi-sample bound (XM's smooth
  form, the GK loss, the deployed per-span Bayesian read) is
  `E log (1/K) ∑ₖ p(y | zₖ)`. The hard min (XM's Forward form) is `E maxₖ log p(y | zₖ)`.
* **Noise.** In `Smooth.lean` and `HardMin.lean` the latent is one real number
  `x = μ + σ n` and the reader is Gaussian, `log p(y|x) = -a (x - y)²` up to a constant, with
  curvature `a = 1/(2s²)`. The target `y` has a finite law.
* **Depth.** In `Hops.lean` the context holds a table `f` (facts, edges) and the target is the
  `h`-th hop `f^[h] x₀`. A computation is an adaptive lookup program: at each step it looks
  the table up at ANY node it chooses, or answers. The view is unrestricted; the number of
  sequential lookups is what is counted.
* **The loop.** In `Washout.lean` a pass is a map with Lipschitz constant `L` (MORPH holds
  the slot loop's typical gain at 0.87 to 0.90). A code enters either once, at the entry, or
  every pass (`h ↦ F h + c`, the way `inj[k]` enters `_tul_core`).

## What is proved

| result | file:line | statement |
|---|---|---|
| **`mixture_le_bayes`** | `TulExploration/Bayes.lean:77` | any random mixture read scores at most `∑ P log P`, the Bayes read |
| **`multisample_le_bayes`** | `Bayes.lean:112` | the `K`-sample bound, any `K`, any sampler, any reader, is at most the Bayes read |
| `multisample_bayes_attained` | `Bayes.lean:132` | a hedging reader attains it with any latent (trivial direction) |
| **`product_reader_le`** | `Bayes.lean:161` | on a perfectly correlated pair, every product reader with any deterministic latent scores at most `-2 log 2` |
| **`enumerated_pair_gt`** | `Bayes.lean:192` | two ENUMERATED codes with committed product readers score `log(((1-ε)²+ε²)/2) > -2 log 2` for every `ε ≠ 1/2` |
| `iid_pair_le` | `Bayes.lean:226` | the same readers fed two IID codes never beat `-2 log 2` |
| **`hardObj_eq`** | `TulExploration/HardMin.lean:148` | hard min, two-point noise: `E max = -aE e² - aσ² + 2a(1-2^{1-K}) σ E\|e\|` exactly, `σ ≥ 0` |
| `hardObj_hasDerivWithinAt` | `HardMin.lean:196` | right derivative at `σ = 0` is `2a(1-2^{1-K}) E\|e\|`: first order |
| `hardObj_le_opt`, `hardObj_escape` | `HardMin.lean:220`, `:237` | `σ* = (1-2^{1-K}) E\|e\|` is optimal and strictly beats `σ = 0` for `K ≥ 2` |
| **`hard_gain_eq_deploy_price`** | `HardMin.lean:256` | the hard-min gain at `σ*` equals the loss of a one-sample deploy at `σ*` |
| `hardObj_one`, `hardObj_useless_spread` | `HardMin.lean:247`, `:269` | one draw: spread is pure price; a target at the latent: spread is pure price |
| `smoothObj_le_common_max`, `_lt_` | `TulExploration/Smooth.lean:110`, `:127` | if one latent maximises the score of EVERY target, it is optimal at every `K`, strictly |
| `single_draw_le_best` | `Smooth.lean:166` | at `K = 1` some deterministic latent is at least as good as any noise |
| `matched_curvature_optimal` | `Smooth.lean:259` | the Gaussian reader's best curvature at `σ = 0` is `a = 1/(2V)` |
| `quadObj_deriv_zero` | `Smooth.lean:328` | `J'(0) = -2a E[μ-y] E[mean w]`, zero for centred draws |
| **`quadObj_deriv2_zero`** | `Smooth.lean:349` | `J''(0) = 4a²V·spread - 2a·m₂` exactly, any finite draw law |
| `quadObj_isLocalMax`, `_isLocalMin` | `Smooth.lean:379`, `:392` | the sign of `2aV·spread - m₂` decides collapse or growth |
| `iid_stats` | `Smooth.lean:481` | `K` iid fair signs: `E mean = 0`, `m₂ = 1`, `spread = 1 - 1/K` |
| `iid_deriv2` | `Smooth.lean:528` | iid: `J''(0) = 2a(2aV(1-1/K) - 1)` |
| **`iid_matched_isLocalMax`**, `iid_matched_strict` | `Smooth.lean:514`, `:618` | matched reader, iid noise: `σ = 0` is a strict local max at EVERY `K ≥ 1` (`J''(0) = -2a/K`) |
| `iid_sharp_not_isLocalMax` | `Smooth.lean:604` | a reader with `2aV(1-1/K) > 1` makes `σ = 0` not a maximiser |
| **`sharp_reader_counterexample`** | `Smooth.lean:639` | targets `±1`, `K = 2`, `a = 2`: `σ = 0` is not a local max. The global T1 claim is false |
| `enumerated_pair_deriv2` | `Smooth.lean:661` | enumerated antithetic pair: `J''(0) = 2a(2aV - 1)`, zero at the matched reader |
| **`hop_lower_bound`** | `TulExploration/Hops.lean:187` | fewer than `h` sequential lookups get the `h`-th hop wrong on some table, with an unrestricted view |
| **`passes_needed`** | `Hops.lean:219` | prelude `Lp`, `T` passes of `b`, coda `Lc` lookups, correct on every table ⇒ `h ≤ Lp + bT + Lc` |
| `pointer_loop_exact` | `Hops.lean:236` | one lookup per pass, `h` passes, is exact: the bound is tight |
| `fixed_after` | `Hops.lean:249` | an absorbing chain end is the loop's fixed point, reached after `h` passes |
| `teacher_forced_one_lookup` | `Hops.lean:260` | given the true previous node, every target is one lookup away (trivial, stated for use) |
| `fixed_table_no_lookups` | `Hops.lean:269` | a relation fixed across inputs is precomposed with zero lookups (trivial, stated for use) |
| `double_iterate` | `Hops.lean:276` | composing two partial results per step reaches hop `2^t` in `t` steps |
| `washout` | `TulExploration/Washout.lean:44` | `dist (Fᵀx) (Fᵀy) ≤ Lᵀ dist x y`: an entry-only code decays |
| `reinjection_separates` | `Washout.lean:56` | re-injected codes keep fixed points `≥ ‖c-c'‖/(1+L)` apart |
| **`reinjected_rollouts_stay_apart`** | `Washout.lean:76` | after `T` passes: separation `≥ ‖c-c'‖/(1+L) - Lᵀ(…)`, a positive floor |
| **`reinjected_linear_diff`**, `entry_linear_diff` | `Washout.lean:132`, `:148` | linear pass: re-injected difference `∑_{t<T} Aᵗ Δc`; entry-only `Aᵀ Δe` |
| **`reinjected_eigen_diff`**, `entry_eigen_diff` | `Washout.lean:167`, `:175` | along a gain-`λ` direction: `(∑_{t<T} λᵗ) Δc` grows with depth; `λᵀ Δe` decays |
| `geom_times` | `Washout.lean:183` | `(1-λ) ∑_{t<T} λᵗ = 1 - λᵀ` |

Helper lemmas (`sum_mul_log_le_log_sum`, `bestOf_eq`, `sum_bestOf`, `q_at_zero`,
`sum_sgn_mul_coord`, `strict_localMax_of_deriv2_neg`, `not_isLocalMax_of_deriv2_pos`, the
`Prog` lemmas, and others) are listed in `Axioms.lean`.

## The theorems in plain English

**T3, width cannot beat the Bayes read (`Bayes.lean`).** Draw any number of latents, in any
correlated way, feed them to any reader, and average the reader's probabilities. The expected
log-likelihood of that mixture is at most that of the true conditional law. The proof is
Jensen (move the draw average inside the log) and Gibbs (no pmf beats the truth). A reader
that can output the true law attains the bound with ONE deterministic latent, or with none.
So width can only matter when the reader class cannot hedge. The smallest case is proved: a
span of two tokens that are always equal. A product reader (a parallel decode from one
latent) scores at most `-2 log 2` whatever deterministic latent it reads; that cap is the
"blur" of a single-vector parallel decode. Two enumerated codes with committed readers beat
the cap for every `ε ≠ 1/2` and approach the Bayes value `-log 2`. Two IID codes with the
same readers never beat the cap: half the draws are collisions. Loses: one context at a time
(the context average is a sum of these statements); the committed-reader example is one
fixed family, not a characterisation of every bounded reader.

**T2, the hard min escapes at first order and pays for it (`HardMin.lean`).** With two-point
noise and a quadratic reader the best-of-`K` objective has an exact closed form. Its slope at
`σ = 0` is positive whenever `K ≥ 2` and the target is not always at the latent. So
`σ = 0` is never its maximiser, as the XM report argued. The optimum `σ*` gains exactly
`a σ*²`, and a deployment that reads one sample without selection loses exactly `a σ*²` at
the same `σ*`. A target that sits at the latent gains nothing and still pays `a σ²` at every
`K`. Loses: symmetric two-point noise (`±1`), not Gaussian noise; one real dimension; the
spread direction is fixed.

**T1, the smooth bound collapses learned noise, but not always (`Smooth.lean`).** The claim
as stated ("for a concave reader the bound is maximised at `σ = 0` for every `K`") is TRUE
when one latent maximises the reader's score for every target, and is proved in that form.
It is FALSE once the target varies: `sharp_reader_counterexample` exhibits a Gaussian reader
sharper than the target spread for which `σ = 0` is not even a local maximum. The corrected
statement is exact to second order. `J''(0) = 4a²V·spread - 2a·m₂`: the mixture's gain and
the curvature's cost are both second order in `σ`. For `K` independent draws
`spread = (1 - 1/K) m₂`, and at the reader's own best curvature `a = 1/(2V)` this gives
`J''(0) = -2a/K < 0`. So `σ = 0` is a strict local maximum at every finite `K`: a learned
noise scale is pushed to zero by the gradient however many samples the bound averages. That
is the GK collapse. The noise grows only if the reader is sharper than
`2aV(1 - 1/K) > 1`, and an enumerated antithetic pair removes the `1/K` shrink, which makes
spread exactly neutral at the matched reader. Loses: one real dimension (for an isotropic quadratic reader, noise along one direction
`u` reduces to this with `μ - y` replaced by its projection on `u`, but that reduction and
any `d`-dimensional statement are not proved here); a Gaussian reader; a local statement (the global landscape can hold a distant
noisy optimum for a sharp reader, which is the bistable case the note discusses); the
reader's curvature is held fixed while `σ` moves (`matched_curvature_optimal` justifies the
matched value at `σ = 0` only).

**T4, depth is necessary only for composition over the context (`Hops.lean`).** A program
may look the table up at any node, adaptively, as often as it likes; only the number of
sequential lookups on its path is counted. If it makes fewer than `h` lookups, it returns the
wrong `h`-th hop on the chain table or on a table that differs from it at one node it never
read. So a forward whose prelude, passes and coda make `Lp`, `b` per pass and `Lc` lookups
needs `Lp + bT + Lc ≥ h`: the loop is necessary exactly when the target's composition depth
exceeds what the fixed path supplies, and it then needs `T ≥ (h - Lp - Lc)/b` passes. The
bound is tight (one lookup per pass, `h` passes). Two remarks make the conditions sharp.
Teacher forcing hands the reader every intermediate node, and then each target is one lookup
deep. A relation that is the same on every input (stored in the weights, or a positional
shortcut such as "the token after position `p`") is a constant the network can precompose
with no lookups. So depth is necessary only for relations that vary with the context, that
the reader's input does not already resolve, and whose depth exceeds the fixed path. No read
restriction appears anywhere in this statement. Loses: the single-thread model counts one
lookup per step; parallel cells that read each other can double their reach per step
(`double_iterate`), and the matching `log₂ h` lower bound for that model is not proved here
(it is Sanford, Hsu and Telgarsky's result for transformers, and it is conditional). The
model says nothing about a network that is a weak composer, which is the likely reason a
plain loop earns on web text without any clean multi-hop structure.

**T5, a contractive loop keeps a choice only if it is re-supplied (`Washout.lean`).** A code
that enters once, at the loop entry, is multiplied by `Lᵀ` and washes out. A code added at
every pass keeps the rollouts' fixed points at least `‖Δc‖/(1+L)` apart, and after `T` passes
the rollouts approach that floor. For a linear pass the difference is exactly
`∑_{t<T} Aᵗ Δc`: every pass adds a term. Along a direction of gain `λ` that is
`(1 - λᵀ)/(1 - λ)` times the one-pass separation, about 4.6 at six passes for `λ = 0.89`.
This is a depth job that exists without any hop structure in the data: the loop integrates a
fixed-size choice, and a reader that needs the modes separated needs the passes. Loses: the
eigen-direction statement is linear; the real core is nonlinear, and only the Lipschitz
bounds (`washout`, `reinjected_rollouts_stay_apart`) hold for it.

## Coarse-level particles (P1 to P6)

The design: run `K` copies of the Parcae spiral testbed's coarse loop (chunks 16) on one
context, each from an independent random start `x⁰ₖ`, and let the coda attend to all `K`
exit states as a set. Exchangeable particles would let `K` be sampled in training and raised
at inference ("train at 4, infer at 128"). The files are `TulExploration/Particles/*.lean`,
imported by `TulExploration/Particles.lean`. Numerics: `numerics/particles_double_well.py`.

| result | file:line | statement |
|---|---|---|
| `particles_spread` | `Particles/Collapse.lean:62` | a `ρ`-Lipschitz pass: pairwise particle distance `≤ ρᵀ δ` (`washout` applied to particles) |
| `particles_to_fixedPoint` | `Collapse.lean:73` | a contraction on a complete space: every particle within `C ρᵀ/(1-ρ)` of the ONE fixed point |
| **`set_read_gap`** | `Collapse.lean:91` | any `Lr`-Lipschitz set read is within `Lr C ρᵀ/(1-ρ)` of the read of `K` copies of the fixed point |
| `mean_read_const` | `Collapse.lean:102` | a mean-pooled read of `K` copies is the read of one |
| **`attn_duplicates`** | `Collapse.lean:119` | softmax attention over `K` copies of a key = ONE copy with logit `+ log K` |
| `attn_duplicates_corrected` | `Collapse.lean:134` | with bias `-log K` on each particle key, `K` copies read as one for every `K` |
| `fixed_points_lip_ge_one`, `no_contraction_two_fixed` | `Collapse.lean:150`, `:158` | two distinct fixed points: every Lipschitz constant on a set holding both is `≥ 1` |
| **`parcae_needs_expansion`** | `Collapse.lean:178` | `x ↦ F(Dx + b)`, `‖Dv‖ ≤ d‖v‖`, two distinct fixed points: `L_F · d ≥ 1` |
| `residual_fixed_points` | `Collapse.lean:191` | MORPH residual form: `‖x-y‖ ≤ ‖D(x-y)‖ + L_u ‖x-y‖` |
| **`stable_pair_forces_expansion`** | `Collapse.lean:228` | 1-D, two locally stable fixed points: `G'(c) > 1` at some `c` between them |
| `iid_drop`, `iid_marginal` | `Particles/IWAE.lean:62`, `:80` | dropping a coordinate of an iid tuple leaves an iid tuple; one coordinate has law `q` |
| **`iwae_mono`**, `iwae_mono_le` | `IWAE.lean:147`, `:209` | iid particles: `E log (1/K)∑ p(zₖ)` is nondecreasing in `K ≥ 1` (Burda et al. 2015) |
| **`iwae_le_log_mean`** | `IWAE.lean:182` | and is `≤ log E_q p` |
| `iwae_mono_avg`, `iwae_avg_le_bayes` | `IWAE.lean:218`, `:228` | averaged over targets; below the Bayes read |
| `concave_mean_mono` | `IWAE.lean:282` | any concave read of the particles' MEAN value is also nondecreasing in `K` |
| **`add_candidate_iff`** | `IWAE.lean:337` | a FIXED (role) candidate raises the uniform mixture iff its likelihood `≥` the current mean |
| `role_cell_can_hurt` | `IWAE.lean:348` | so a worse role cell strictly lowers the read |
| `mixture_le_best`, **`best_le_mixture_add_logK`** | `Particles/SetRead.lean:50`, `:62` | span oracle minus uniform mixture lies in `[0, log K]`, per span and draw |
| `post_sum_one`, `post_past_only` | `SetRead.lean:95`, `:101` | posterior weights over particles sum to one and use only tokens already seen |
| **`seq_bayes_eq_span_mixture`** | `SetRead.lean:122` | the token-by-token posterior read EQUALS the span mixture (chain rule) |
| `seqRead_le_token_oracle` | `SetRead.lean:138` | any past-only weighting is at most the per-token oracle |
| **`switching_beats_span_oracle`** | `SetRead.lean:167` | a past-only switching read scores `0` where the span oracle and the mixture score `log ε` |
| `arith_pool_floor` | `SetRead.lean:196` | a probability mixture keeps the floor `w_k p_k(y)` |
| **`value_mix_loses_floor`** | `SetRead.lean:210` | value mixing + linear head: read `≤ (c-d)/2` (to `-∞`) while the probability mixture `≥ -log 4` |
| `committee_can_beat_mixture` | `SetRead.lean:253` | and the reverse: errors that cancel give the value mix `1/3`, the probability mix `≤ e^{-M}` |
| `mixture_grad_resp` | `Particles/Credit.lean:57` | the soft objective's derivative along particle `k` is its responsibility `r_k` |
| **`resp_tendsto_one`**, `resp_tendsto_zero` | `Credit.lean:114`, `:95` | at zero temperature the strict winner gets all the credit: WTA is the limit |
| **`soft_grad_at_collapse`** | `Credit.lean:134` | at a collapsed configuration every particle gets the same gradient `ℓ'/K` |
| **`soft_split_collapses`**, `soft_split_grows` | `Credit.lean:164`, `:173` | split `μ ± η`, soft read: `η = 0` local max if `2aV < 1`, local min if `2aV > 1` |
| **`hardPair_eq`**, `hardPair_escape` | `Credit.lean:197`, `:213` | WTA read of the split: `J(0) + a(2ηE|μ-y| - η²)`, first-order escape at every `a > 0` |
| **`colip_lower`**, `passes_to_settle` | `Particles/Depth.lean:55`, `:70` | co-Lipschitz rate `λ`: `T ≥ log(d₀/ε)/log(1/λ)` passes to reach `ε` of the basin point |
| **`saddle_read_margin`**, `passes_to_decide` | `Depth.lean:83`, `:95` | expansion `μ` at the boundary: a Lipschitz reader needs `T ≥ log(m/(Lr δ))/log μ` passes to decide a particle that started `δ` from the boundary |
| **`monotone_side_preserved`** | `Depth.lean:112` | 1-D increasing pass: a particle never changes side of the boundary |

**P1, a contractive pass collapses the particles, and plain attention counts the copies.**
If every coarse pass is a contraction, the `K` particles converge to one fixed point at
rate `ρᵀ`, and any Lipschitz set read approaches the read of `K` copies of that point. `K`
copies are not the read of one particle under plain softmax attention: they act as one key
with its logit raised by `log K`. Raising `K` from 4 to 128 at inference therefore adds
`log 32 ≈ 3.47` to the coarse memory's logit even when the particles carry nothing new. A
`-log K` bias on every particle key removes that and makes `K` collapsed particles read as
one. (For distinct particles that bias makes the group a softmax within the group mixed in
as one key of logit `log mean exp sₖ`; that identity is algebra and is not proved here.) Loses: the
read is held fixed; the contraction must hold on the whole space (on an invariant complete
subset the same proof works but is not restated).

**P2, basins need a non-contractive pass.** Two distinct fixed points force the pass's
Lipschitz constant on a set holding both to be at least 1, so per-context diversity at the
fixed points rules out a contraction on the region the starts cover. For the Parcae form
`x ↦ F(D⊙x + b)` the condition is `L_F · d ≥ 1`, with `d` the largest decay
`exp(-ΔᵢAᵢ)` along the inter-basin direction: the blocks must expand by `1/d`. In 1-D, two
locally stable fixed points force a point between them with local gain above 1 (the
saddle). The connection to MapCause's 0.865 floor is weak and stated as such: that number is
the TYPICAL (Frobenius) gain of MORPH's residual `DiagonalInjection`, where 704 of 1024
channels have decay 1. In the residual form `residual_fixed_points` asks nothing of the
blocks along an undecayed channel and `L_u ≥ 1 - 0.447` along the context slice. No Parcae
testbed decay value is read here, so no number transfers to the coarse loop.

**P3, train at 4, infer at 128, holds for exchangeable particles and fails for role cells.**
With iid particles and a fixed reader, the expected uniform-mixture read is nondecreasing
in `K` and bounded by `log E_q p(y|z)` (and so by the Bayes read). The proof uses
exchangeability: each leave-one-out `K`-subset of `K+1` iid particles is again `K` iid
particles. For fixed role-indexed candidates there is no such statement: a new candidate
helps iff its likelihood is at least the current mean, so a worse cell lowers the read. That
is the shape of the variable-cell-count v1 failure (fresh or cyclic extra cells cost +0.05
to +0.14 nats). The same monotonicity holds for any concave read of the particles' mean
value, which covers uniform-weight attention with a linear head, but its limit is the read
of the mean particle, not the mixture. Loses: the reader is fixed while `K` changes; the
limit `K → ∞` (law of large numbers) is not formalised; real attention weights depend on the
particles, so the read is a ratio of two sample means, and no monotonicity is proved for it.

**P4, the `log K` ceiling and the sequential Bayes read.** Per span and per draw the best
particle beats the uniform mixture by at most `log K`. Reading the span token by token, each
token from the posterior over particles given the true prefix, is EXACTLY the span mixture
(the chain rule telescopes), and those weights use only tokens already seen. So a mixture
objective in training and a posterior-weighted set read at deploy are the same objective.
Two things break the simple picture. First, a past-only reader that switches particles
between tokens can beat the span oracle, per draw, by an unbounded amount: the claim "a
learned past-only weighting lies between the mixture and the oracle" is false. (In
expectation over the true law a past-only weighting still defines a pmf over spans and stays
below the Bayes read; that is not formalised.) Second, attention mixes VALUES, not
probabilities. With a linear head that is a normalised geometric pool, which has no `log K`
floor: one confidently wrong particle drives it to `-∞` while the probability mixture stays
above `-log 4`. It is not uniformly worse either: errors that cancel in the mean favour it.
Loses: the coda has many layers after the attention, so "linear head" is the one-layer case.

**P5, soft credit vs winner-take-all.** The soft objective credits particle `k` with its
responsibility; with temperature `1/β → 0` the strict winner's responsibility tends to 1 and
the losers' to 0, so WTA is the zero-temperature limit. At a collapsed configuration the soft
objective gives every particle the same gradient, so soft credit cannot break a collapse at
first order. For a symmetric split `μ ± η` with the Gaussian reader the soft objective
holds the collapse (`η = 0` a local max) when `2aV < 1` and splits it when `2aV > 1`; WTA
gains `2aηE|μ-y|` at first order, so it splits at every sharpness. "Soft collapses, WTA keeps
basins apart" is therefore true for a broad reader and false as a general claim. For
particles from iid random starts, the soft threshold is the stricter `2aV(1-1/K) > 1`
(`iid_deriv2`): at the matched reader the soft objective pushes the particles' start
sensitivity `σ` to zero at every finite `K`, which is `Smooth.lean`'s T1 collapse. Loses: one
real dimension; a Gaussian reader; local (second-order) statements; the particle positions
are treated as the free variable, while in the design they are outputs of a shared map.

**P6, what depth buys.** No depth lower bound holds for a general pass (a pass that maps
every start to its basin point needs one pass). Two local hypotheses give one. A
co-Lipschitz rate `λ` toward the basin point gives `T ≥ log(d₀/ε)/log(1/λ)`. A Lipschitz
rate `μ > 1` at the basin boundary gives: a particle that starts `δ` from the boundary reads,
to any `Lr`-Lipschitz reader, like the boundary point until `T ≥ log(m/(Lr δ))/log μ`. In 1-D
with an increasing pass the particle never changes side, so basin IDENTITY is fixed by the
start and depth buys only MARGIN. Numerics (`particles_double_well.py`, `tanh(2x) + c`,
starts `N(0,1)`, 200k particles): the undecided fraction `|x_T - x_u| < 0.25` falls
0.197, 0.102, 0.051, 0.026, 0.013, 0.0063, 0.0032 over `T = 0..6`, within 4 % of the
`μ^{-T}` prediction (`μ = 2`; the gap grows to 20 % by `T = 11`); no particle changed
side; the region-wide co-Lipschitz bound held for every particle that started in
`(x_u, 3]` (rate 0.021, so the bound is weak), while the LOCAL rate `G'(x*)`
is not a valid bound (16 % of particles settle faster, saturation carries far starts in one
pass). The basin set depends on the context `c`: two stable points for `|c| < 0.266`, one
beyond; at `|c| = 0.3` and `T = 8` the particles still split 31/69 % by slow passage past
the vanished basin, a depth-dependent transient, not an attractor. A contracting control
(`β = 0.6`) has spread ratio below `ρᵀ` at every `T`. Loses: 1-D; the fraction-of-starts
statement is numerical, not a theorem.

## What is assumed, and what is NOT proved here

* **Finite laws and one context at a time.** Every statement is per context; averaging over
  contexts is a sum and is not restated.
* **Positive readers.** Readers are strictly positive pmfs (softmax), which the Jensen step
  needs.
* **The bridge to MORPH's code is by description, not by proof.** Nothing here reads
  `morph/model/transformer.py`. That the TF coda is a hedging reader, that one core block is
  about one lookup, and that the measured gain `λ ≈ 0.89` is the gain along a code direction
  are modelling claims in the note, not theorems.
* **No estimation theory.** Nothing bounds what a TRAINED network reaches in 5,000 steps.
* **The T1 second-order statements are local.** They say where the gradient pushes a small
  `σ`, not where a large one ends up.
