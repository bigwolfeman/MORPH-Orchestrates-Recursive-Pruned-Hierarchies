# Agent Note: Provable loop contribution: what width and depth need, and one slot-loop design that meets it

Status: proposed

## Problem

The slot loop does not earn. Twelve designs without a forced relay read token K1−K6 in
[−0.0001, +0.0033]
([`../../implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md`](../../implemented/architecture/2026-09-22-slot-loop-campaign-synthesis.md)).
The two stochastic arms of 2026-09-23 failed in opposite ways. LXTUL-G trained its coda on
posterior cells and deployed prior cells: exposure gap 0.9840 at 5000
([`../../../../lab/experiments/failures/2026-09-23-lxtul-g-panel.md`](../../../../lab/experiments/failures/2026-09-23-lxtul-g-panel.md)).
LXTUL-GK trained on the deployed rollouts under the multi-sample bound, and its learned noise
died: on `lxtul-gk4-shared` prior sigma/r 0.00026, eval width gain at four samples 0.00005,
K1−K6 +0.0036; on `lxtul-gk1` sigma/r 0.000339, final val 4.4238 against the ruler's 4.4249
([`../../../../lab/experiments/failures/2026-09-23-lxtul-gk-multisample.md`](../../../../lab/experiments/failures/2026-09-23-lxtul-gk-multisample.md)).

Wolfe (2026-09-23): "we need to actually prove and design a real solution. Maybe in lean."
The existing Lean development
([`../../../../lab/theory/tul_information/`](../../../../lab/theory/tul_information/),
note [`2026-09-13-information-view-of-the-slot-loop.md`](2026-09-13-information-view-of-the-slot-loop.md))
proves that a deterministic pass adds no information, so the loop can earn only by a relay
or by extractability. It says nothing about width, and nothing about when depth is
NECESSARY. This note closes those two gaps with proofs and derives one design from them.
Standing constraints: no read restriction geometry, depth use emergent from a mostly
contractive loop, web text with no task reward, a teacher-forced AR coda that sees the true
prefix.

## Proposal

### What is proved

Lean 4, Mathlib v4.31.0, in
[`lab/theory/tul_exploration/`](../../../../lab/theory/tul_exploration/). `lake build` exits
0 with zero warnings (2195 jobs). `lake env lean Axioms.lean` prints 95 lines: 80 read
`[propext, Classical.choice, Quot.sound]`, 15 use a subset of those, none mentions
`sorryAx`. The README gives file:line for every result and what each simplification loses.

**T3, width cannot beat the Bayes read** (`TulExploration/Bayes.lean`).
`mixture_le_bayes` and `multisample_le_bayes`: for any `K`, any way of drawing the `K`
latents, and any positive reader, the expected log of the averaged read is at most the
log-likelihood of the true conditional law. `multisample_bayes_attained`: a reader that can
output the true law attains it with one deterministic latent. So width pays only when the
reader class cannot hedge. `product_reader_le`: on a two-token span whose tokens are always
equal, every product reader (a parallel decode from one latent) with any deterministic latent
is capped at `-2 log 2`. `enumerated_pair_gt`: two enumerated codes with committed product
readers beat that cap for every `ε ≠ 1/2`. `iid_pair_le`: two iid codes with the same
readers never beat it.

**T2, the hard min escapes and pays** (`TulExploration/HardMin.lean`). `hardObj_eq`: with
two-point noise and a quadratic reader the best-of-`K` objective is exactly
`-aE e² - aσ² + 2a(1 - 2^{1-K}) σ E|e|`. `hardObj_hasDerivWithinAt`, `hardObj_escape`: its
slope at `σ = 0` is positive for `K ≥ 2`, so the hard min never collapses the noise.
`hard_gain_eq_deploy_price`: the gain at the optimal spread equals, to the digit, what a
one-sample deployment loses at that spread. `hardObj_useless_spread`: a target already at the
latent pays `aσ²` for nothing.

**T1, the smooth bound collapses learned noise, and the brief's statement of it is false**
(`TulExploration/Smooth.lean`). As asked ("for a concave reader the bound is maximised at
`σ = 0` for every `K`") it holds only when one latent maximises the reader's score for every
target (`smoothObj_le_common_max`, `smoothObj_lt_common_max`). Once the target varies it is
false: `sharp_reader_counterexample` gives a Gaussian reader for which `σ = 0` is not a local
maximum. The corrected statement is exact: `quadObj_deriv2_zero`,
`J''(0) = 4a²V·spread - 2a·m₂`. Gain and cost are both second order in `σ`. For `K` iid draws
`spread = (1 - 1/K)` (`iid_stats`), so at the reader's own best curvature `a = 1/(2V)`
(`matched_curvature_optimal`) the curvature is `-2a/K` and `σ = 0` is a strict local maximum
at EVERY `K` (`iid_matched_isLocalMax`, `iid_matched_strict`). Noise grows only for a reader
sharper than `2aV(1 - 1/K) > 1` (`iid_sharp_not_isLocalMax`). An enumerated antithetic pair
removes the `1/K` shrink and is exactly neutral at the matched reader
(`enumerated_pair_deriv2`).

**T4, depth is necessary only for composition over the context** (`TulExploration/Hops.lean`).
A program may look up a context table at any node, adaptively; only the number of sequential
lookups is counted. `hop_lower_bound`: fewer than `h` lookups get the `h`-th hop wrong on some
table, although the view is unrestricted. `passes_needed`: a prelude of `Lp`, `T` passes of
`b`, and a coda of `Lc` lookups that is right on every table has `h ≤ Lp + bT + Lc`.
`pointer_loop_exact`: the bound is tight. `fixed_after`: a chain that ends in an absorbing node
makes the answer the loop's fixed point, reached in `h` passes. `teacher_forced_one_lookup`
and `fixed_table_no_lookups` (both trivial, stated so the note can cite them): the true
previous node makes every target one lookup deep, and a relation that is the same on every
input needs no lookups at all. `double_iterate`: parallel cells that compose each other's
pointers reach hop `2^t` in `t` steps; the matching lower bound is not proved.

**T5, a contractive loop keeps a choice only if it is re-supplied**
(`TulExploration/Washout.lean`). `washout`: a code that enters once is multiplied by `Lᵀ`.
`reinjection_separates`, `reinjected_rollouts_stay_apart`: a code added at every pass keeps
the rollouts at least `‖Δc‖/(1+L)` apart after a transient that decays like `Lᵀ`.
`reinjected_linear_diff`, `reinjected_eigen_diff`: for a linear pass the difference is
`∑_{t<T} Aᵗ Δc`, which along a direction of gain `λ` is `(1 - λᵀ)/(1 - λ)` times the one-pass
separation (`geom_times`). At the measured typical gain 0.89 that is 4.6 at six passes.
`entry_eigen_diff`: the entry-only code gives `λᵀ` instead.

### How the ledger reads under these theorems

Each line is a reading, not a new measurement.

* GK's collapse is `iid_matched_strict`: a coda trained at small `σ` sits near its matched
  curvature, and there the gradient pushes `σ` to zero at every `K`.
* GK1 tying the ruler is `multisample_le_bayes` with `multisample_bayes_attained`: the
  teacher-forced coda hedges, so a deterministic cell is already optimal for it.
* The fan's streams collapsing to copies (register rank 1.24 of 4, cos 0.99) is the same
  statement for width without noise: copies are optimal for a hedging reader.
* "Every per-pass target is met in one step" for the M-next and bag targets is, I infer, the
  product cap: a committed reader with one deterministic latent can reach only the product of
  marginals (`product_reader_le`), and marginals are shallow. This is an interpretation; no
  file measured it.
* "The loop contracts entry noise" (memory, fp0 and noise-entry arms) is `washout`.
* The slot loop's token K-curve staying flat under a teacher-forced coda is
  `teacher_forced_one_lookup` beside `Joint.mi_relay_redundant`: the coda's targets are one
  lookup from what it already sees.
* The plain loop's 0.1849 is not explained by T4's ideal composer. A network is a weak
  composer, so extra blocks on every token help without any clean multi-hop structure. T4
  only says why the slot loop cannot collect that gain: its passes deepen the cell, not the
  token path the coda is charged on.

### The requirements

(a) Width survives training and helps only if all four hold.

* W1, a committed reader. The read that pays for width cannot be able to output the
  conditional law (`multisample_le_bayes`, `multisample_bayes_attained`,
  `enumerated_pair_gt`). The teacher-forced coda is not such a reader.
* W2, no learnable scale under the smooth bound. A learned `σ` dies at every `K`
  (`iid_matched_strict`). The choice must be a discrete code with a fixed size.
* W3, enumerate, do not sample. Iid collisions can erase the gain at small `K`
  (`iid_pair_le`); enumeration turns the bound into the exact mixture likelihood and removes
  the `1/K` shrink (`enumerated_pair_deriv2`). With enumeration and a committed reader the
  hard min is not needed, and its gain is a deploy price anyway (`hard_gain_eq_deploy_price`).
* W4, re-supply the choice every pass. An entry-only choice washes out in a contractive loop
  (`washout`, `entry_eigen_diff`).

(b) Depth is necessary, with nothing restricted, only if all three hold.

* D1, the target composes relations that vary with the context, deeper than the fixed path's
  lookups `Lp + Lc` (`passes_needed`, `fixed_table_no_lookups`).
* D2, the read does not already hold the intermediate results (`teacher_forced_one_lookup`).
* D3, the per-pass budget `b` is small against `h - Lp - Lc`, or pass 1 does it all
  (`passes_needed`).

T5 adds a second kind of depth job that does not need D1: integration. A choice re-injected
at a fixed size is amplified by `∑ λᵗ` over the passes (`reinjected_eigen_diff`), so a reader
that needs the modes separated beyond one pass's `‖Δc‖` needs the passes. That job exists
only if W1 holds, because only a committed reader pays for separation.

### The design: LXTUL-E, an enumerated choice integrated by the loop and read in parallel

```
LXTUL-E (one slot loop, K = 4 rollouts per row, all enumerated)
├── choice    code k in {1..4}: a learned unit direction u_k, mean-free across k (W2)
│             added EVERY pass at fixed size: h ← f(h) + r · rms(h).detach() · u_k,
│             r = 0.1 (W4, T5); no σ, no sampling (W3)
├── loop      the ruler's core, fixed-point term 1.0, gain constraint on: contractive
│             passes must integrate the choice (separation ∝ Σ λᵗ) and settle each
│             rollout at its own fixed point, one continuation mode per code
├── readers
│   ├── coda        unchanged: reads every cell and every token, teacher forced;
│   │               token CE under the EXACT per-span mixture over the 4 rollouts
│   │               (GK's `iw` read; exact because the codes are enumerated)
│   └── parallel    replaces the teacher-forced span decoder: all tokens of the next
│                   span predicted at once from z_k and a position query, no token
│                   input (W1, D2); loss −log (1/4) Σ_k Π_j p(t_j | z_k, j);
│                   tied embeddings detached; weight 1.0 like `spandec_weight`
└── deploy    the coda's per-span Bayesian read over the 4 enumerated rollouts
```

What each part answers:

* The objective is the exact mixture likelihood of two readers. It is a proper score, so it
  cannot reward spread the data does not have (`mixture_le_bayes`), unlike the hard min or a
  diversity term.
* The latent is a deterministic trajectory per code. The only choice is which code, and all
  four are always evaluated. Nothing can shrink it.
* The coda reads what it reads today. No geometry changes.
* The passes have one job the parallel read pays for (integrate the code into a separated
  mode, T5) and, where the next span composes context relations, a second one (D1).

Build notes: the rollout expansion, shared dropout masks (`morph/model/rollout_dropout.py`,
7d44ed7) and the per-span mixture read already exist for GK. New: the code table and its
per-pass injection, and a `tul.spandec_parallel` path in `morph/model/tul_spandec.py`
(position queries instead of token inputs). Proposed keys `tul.code_enum_k`,
`tul.code_enum_ratio`, `tul.spandec_parallel` must enter `KNOWN_TUL_KEYS`.

### Which requirement web text may not satisfy

D1. T4's necessity needs relations that vary with the context, have no positional or
memorised shortcut, and are deeper than the fixed path's lookups. Web text has such
dependencies (entity tracking, coreference chains, variable flow in code), but few of them
are deeper than the six prelude and coda blocks, and fewer still resolve at the span level
where the slot loop works. W1 is the second risk: the next span may carry little
multimodal joint structure that four codes can capture once the cell is given. Stage 0
below measures that before any loop is trained.

What it would mean. If both fail, no objective on the slot channel can make depth necessary
while the coda is teacher forced: the coda's own targets are one lookup deep, and the
parallel read would have nothing to integrate. That is not "web text does not need depth".
The plain loop earns 0.1849 on web text, so the depth web text rewards is on the token path,
which the slot loop does not run. The job would then have to come from data with deep
context composition, or from giving the passes the context to compose (the reread arm,
which strict geometry forbids as a bypass).

### The cheapest decisive test

**Stage 0, no loop training, about one GPU-hour.** Load the kept `slot-spandec-strict`
checkpoint at step 5000 and freeze it. Train two parallel heads on its exit cells for 2000
steps on the arc's rows: one reads the cell alone, one reads the cell plus the four
enumerated codes at the head input under the mixture loss. Score both on the 480 validation
rows. The width budget is `B0 = CE_par(one) − CE_par(four-code mixture)` per span token,
bucketed by position in the span. By `multisample_le_bayes`, width cannot gain more than the
reader's distance to the Bayes read given the cell, and `B0` measures how much of that four
codes reach from the ruler's cells.

**Stage 1, two arms of 5000 steps** on the strict recipe (seq 1024, batch 6, seed 1,
norm_match, ramp 1000): `lxtul-e4` (the design) and `lxtul-e1` (the same parallel head and
loop, `K = 1`, no code). One factor: the enumerated re-injected choice. Partners on disk: the
ruler and `lxtul-gk1` at 5000.

## Alternatives considered

* **Keep GK's learned Gaussian step and raise `K`.** Rejected by `iid_matched_strict`: the
  collapse holds at every finite `K`, and the threshold factor `1 - 1/K` never reaches one.
  `lxtul-gk4-shared` read sigma/r 0.00026 (P-0 below).
* **XM's hard min with fixed noise.** Considered first, because it is the one objective the
  XM report found that does not collapse. `hardObj_escape` confirms the escape, but
  `hard_gain_eq_deploy_price` shows the gain is paid back in full by any deploy that reads one
  sample, and `hardObj_useless_spread` charges spread on targets that do not need it. With
  enumeration and a committed reader the exact mixture already pays for distinct codes, so
  the hard min adds a price and no benefit.
* **A diversity term (Bifurcation, the fan's epivol).** Rejected for the same reason as the
  hard min: it pays for spread whether or not a reader uses it. The fan measured the
  consequence (rank held, reader cashed little).
* **Deterministic superposition only, no width (the RbS view).** It is optimal for the
  teacher-forced coda (`multisample_bayes_attained`) and that is why it cannot be the lever:
  it changes nothing the coda lacks. Measured the same evening
  ([`../../../../lab/experiments/successes/2026-09-23-superposition-probe.md`](../../../../lab/experiments/successes/2026-09-23-superposition-probe.md)):
  the ruler's cell is already uncommitted. With the document held fixed it helps the
  continuation 1.74x as much when the context's runner-up first token came true, and GK's
  cells read the same. That is the hedging read this theorem describes. For a committed reader one deterministic latent is
  capped at the product of marginals (`product_reader_le`).
* **The choice at the loop entry only (the fan's streams, noise entry).** Rejected by
  `washout` and `entry_eigen_diff`: the loop contracts it, as measured.
* **Thin passes on the ruler (`n_core 2`, mean 18, the superposition report's R3).** Kept as
  the follow-up depth factor, not the first test. With a teacher-forced coda D2 fails, so
  `passes_needed` predicts the same work spread over more passes: dependence without value.
* **The reread (cells read the prelude tokens every pass).** It supplies the context D1 needs
  and is the RbS "re-read the edge list" step. Rejected for the first test: strict geometry
  forbids it as a cross-span bypass, and it read flat (0.0003) with a one-latent target.
* **A read restriction (`prev-reach1`).** Excluded by Wolfe. `hop_lower_bound` shows it is not
  needed in principle: necessity comes from composition depth with an unrestricted view.
* **Posterior plus KL (LXTUL-G), or a forward-KL posterior.** Rejected. `mixture_le_bayes`
  bounds any posterior-trained mixture the coda reads, and the measured exposure gap grew with
  training.

## Acceptance criteria

Frozen here, before any build. Probabilities are my priors. No clause passes or fails on a
cosine.

* **P-0 (T1 on the pending run).** `lxtul-gk4-shared` at 5000 has prior sigma/r below 1e-3
  and eval `width_gain@4` below 0.001 nats: **80 %**. Amendment at merge (2026-09-23 21:50):
  the run was filed at 21:24, before this note was merged, so P-0 is a reading, not a
  prediction: sigma/r 0.00026, width gain 0.00005. Consistent with T1. If sigma/r stays above 0.01 with a
  width gain above 0.01, the coda is sharper than the matched curvature
  (`iid_sharp_not_isLocalMax`), and that is worth measuring before LXTUL-E.
* **P-S0 (the width budget).** `B0 ≥ 0.02` nats per span token: **55 %**. `B0 < 0.005`
  stops the design: web text's next span, given the ruler's cell, has no joint structure four
  codes can use. Between the two, Stage 1 runs with its width priors halved.
* **P-1 (width survives).** `lxtul-e4`'s parallel mixture CE beats `lxtul-e1`'s by at least
  `max(0.01, 0.5 B0)` per token, paired CI clear of zero: **50 %**.
* **P-2 (the coda hedges).** `lxtul-e4`'s token width gain (one code alone minus the
  four-rollout read) is at most 0.005: **85 %**. Above 0.01 means the coda has an inference
  deficit that width fills, an extractability result.
* **P-3 (width creates a depth job).** `lxtul-e4`'s parallel-read K1−K6 is at least 0.02 with
  the CI clear of zero: **35 %**. `lxtul-e1`'s parallel-read K1−K6 is at most 0.005: **60 %**.
  Diagnostic beside it: the exit separation between codes at forced depth 6 over depth 1
  (linear theory: 4.6 at gain 0.89).
* **P-4 (the teacher-forced bypass).** Token K1−K6 of the coda is at most 0.005 on both arms:
  **80 %**.
* **P-5 (no price).** `lxtul-e4`'s token CE under the four-rollout read, paired against the
  ruler at depth 6, is within +0.02: **55 %**.
* **What refutes the account.** P-1 holds, P-3's first clause fails, and the separation ratio
  is below 1.3: the reader reads the one-pass separation directly and integration is not a
  job. Or P-2 fails by more than 0.02 with P-1 failing: the teacher-forced coda is not the
  hedging reader T3 treats it as, and the bridge from the theorems to the coda is wrong.

## Risks

* **The parallel head is the thing CLAUDE.md warns against**: a span decoded from one vector
  plus an offset with no token path (Huginn, MegaByte, Bowman, Hourglass). Here it is a
  training target, never the decoder; the coda keeps the token path. `product_reader_le` is
  the proof of that warning (one vector plus a product read is capped at the blurred
  marginals), and the enumerated mixture is the remedy it implies. The risk that remains is
  that its gradient drags the cell toward a form the coda reads worse; P-5 watches it.
* **Scale defeats integration.** The parallel head can scale its weights to read a small
  separation, and then T5's job is not needed. Fixing `r` does not stop that.
* **The theory is idealised.** T1 and T2 are one-dimensional with a Gaussian reader. T5's
  amplification is linear. T4 counts ideal lookups; a real block is a weak composer, which is
  how the plain loop earns without clean hop structure.
* **`B0` is measured on the ruler's cells.** A trained LXTUL-E loop could put more joint
  structure into its cells than the ruler did, so a small `B0` is evidence, not proof.
* **Cost.** Four rollouts cost what GK4 costs: a traced 15.17 GB step and the recompute of the
  coda and span decode (the GK prereg's amendment).
* **One seed, 5000 steps.** Short-horizon CE does not rank arms. Every clause above is a
  within-run K-curve or a paired gap on identical rows.
