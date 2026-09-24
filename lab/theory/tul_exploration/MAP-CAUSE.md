# What sets the loop's per-pass map: the cause behind 0.88

Date: 2026-09-24. Lean file: [`TulExploration/MapCause.lean`](TulExploration/MapCause.lean) (T6).
Toy scripts (`toy_map.py`, `linear_checks.py`, `readA.py`, `wnorms.py`) are in
[`numerics/`](numerics/); the toy's per-run JSON and logs are in
[`numerics/toy/`](numerics/toy/). The numbers that matter are copied below.

## The question

The plain loop's per-pass map reads a typical gain near 1 (0.97 to 0.92 on passes 1 to 5),
varies by position and has a tail above 1. The slot loop's reads a uniform 0.87 to 0.88 with
no reading above 1, and the coda gets no depth value from it. Wolfe: the map is an effect.
What cause yields it?

## Short answer

1. **The number 0.865 is the injection's decay, not a choice the loss made.** A pass whose
   blocks do not respond to the state has a typical gain set by `DiagonalInjection` alone:
   identity on 704 of 1024 channels, decay `A` on the 320 context channels. With the learned
   `A` read from the checkpoints this floor is 0.864 to 0.867 in every model. The slot map sits
   0.005 to 0.018 above it. The plain map sits 0.05 to 0.10 above it (passes 1 to 5) and 0.38
   above it on pass 0. So the slot core's blocks barely respond to the slot state, and the
   plain core's blocks do.
2. **The loss pushes the gain toward 1 only on directions that must KEEP work across passes.**
   Proved in a linear model: when every pass brings new evidence that the reader wants
   summed, the only exact fit is gain 1 and the loss falls strictly toward 1. When the source
   is the same every pass, a depth-independent target read at a random depth is fit only by a
   one-step map or by an entry already at the fixed point, and along the second family the
   loss and the fixed-point term cannot see the gain at all. A relay (move information, forget
   the last hop) asks for gain 0.
3. **On a settled trajectory the gradient cannot tell the recurrence from a per-pass write.**
   Proved: the derivative of the exit state along a change `G` of the recurrence splits into
   the derivative of a write of `G h*` plus a recurrence-only term bounded by the transient
   `‖h₀ - h*‖`. At `h₀ = h*` that term is exactly zero. The slot state settles by pass 3 with a
   norm ratio of 1.02 then 1.006. The plain state grows 6.2x on pass 0.

4. **The fixed-point term squeezes a WORKING loop's map into a flat contraction** (trained
   toy, two seeds), without removing its depth value. So a flat map alone does not prove
   "no work". The slot loop's lack of work is read from three things that do not depend on
   the map: the coda K-curve (0.001 to 0.002), a pass 0 as flat as pass 5, and a state that
   barely moves.

A fact the brief states differently: `plain-panel-norm-match` composes with
`core_state_init: prelude`, so its entry is `h₀ = input_norm(x)`, not noise. It is small
relative to the state the loop builds (x6.2 on pass 0), which is what matters here.

The causal chain for the slot loop, as I read it: the coda has its own in-span token path,
and the cross-span content it needs arrives in one attention pass over earlier slots. So the
slot trajectory has nothing to keep and nothing to build. It settles at once. On a settled
trajectory the task gradient on the recurrence equals a write's gradient, which the prelude
already supplies. Nothing trains the blocks' response to the slot state, and the map reads
the injection's floor. The plain loop is the opposite on each link: the coda reads each
position's own looped state as its main computation, the state is built over passes from an
entry about 6x smaller than its working scale, and each pass keeps and extends the last.

## What was measured here (not proved, not inferred)

Read on CPU from the step checkpoints with `torch.load`, no forward pass. Floor
`= sqrt((704 + Σ A²) / 1024)`.

| model | mean `A` (init 0.447) | mean `dt` (init 1) | floor | measured typical gain (from the brief) |
|---|---|---|---|---|
| plain-panel-norm-match @5000 | 0.451 | 1.001 | 0.867 | pass 0 1.25; passes 1 to 5 0.97 to 0.92 |
| slot-spandec-strict @5000 (ruler) | 0.439 | 0.980 | 0.865 | 0.877 to 0.883 |
| lxtul-e4 @5000 | 0.444 | 0.998 | 0.865 | 0.877 to 0.883 |
| lxtul-e4probe @5000 | 0.433 | 0.970 | 0.864 | 0.869 |
| lxtul-e1 @5000 | 0.438 | 0.990 | 0.865 | not measured |
| lxtul-e4j4 @5000 | 0.444 | 1.023 | 0.866 | not measured |
| slot-spandec-strict-prev-reach1 @5000 | 0.432 | 0.962 | 0.864 | not measured |
| slot-spandec-strict-fan4-all @5000 | 0.478 | 1.065 | 0.871 | not measured |
| slot-spandec-strict-20k @20000 | 0.435 | 0.966 | 0.864 | not measured |
| strict-ruler-50k @50000 | 0.431 | 0.952 | 0.863 | not measured |

`A` has not moved from its init in any model, including the plain one and a 50k-step run.
Nothing in any of these losses pushes on the injection decay.

The core's weights are not small in the slot model. Per-element RMS of the core's matrices,
plain vs ruler: attention `W_up` 0.0256 vs 0.0259, `W_v_curr` 0.0181 vs 0.0177, MLP
`gate_up` 0.0207 vs 0.0179, `down` 0.0197 vs 0.0184. So the slot core is quiet because of
where it operates, not because its weights decayed. Weight decay could not do it anyway: lr
1e-4 times wd 0.1 shrinks a weight by about 5 % in 5000 steps. Which mechanism makes the
blocks insensitive at the slot operating point (saturated gates, focused attention,
pre-norm scaling) is NOT measured.

Two caveats on the floor arithmetic. It assumes the finite-difference perturbation is
isotropic over the `n × C` carrier and that the Hyper-Connection stream mixer `H_res` is
orthogonal (it is, by the Cayley construction), so the mixer does not change the typical
gain. And a cross term between `D` and the blocks' Jacobian can push a reading below the
floor: the toy's settled maps read 0.02 to 0.05 below it.

## The theorems (T6, `MapCause.lean`)

`lake build` exits 0, "Build completed successfully (2196 jobs)", no warnings.
`lake env lean Axioms.lean` prints 123 lines, 0 mention `sorryAx`, 110 end
`[propext, Classical.choice, Quot.sound]` (the rest use a subset). The 28 new lines are the
`-- T6` block at the end of `Axioms.lean`.

Build note: in the main checkout, `lean` blocked in `__fuse_simple_request` for more than
ten minutes with zero CPU time (the HDD's FUSE mount). I built in a mirror at
`/home/wolfe/lean-build/work/tul_exploration` (the same sources copied over, `.lake`
symlinked to the same build directory). Nothing else differs.

### 1. The floor

* `gain_sandwich`. For a pass `F h = D h + u h + c` with `u` `L`-Lipschitz, the finite
  difference along `d` lies in `[‖D d‖ - L‖d‖, ‖D d‖ + L‖d‖]`. A silent core reads the
  injection.
* `silent_gain_le`. A non-expansive `D` (the injection clamps `A ≤ 0.9999`) gives no
  direction above `1 + L`. Zero readings above 1 is what a quiet core looks like.
* `diag_sq_le`, `injection_frob`. The concrete diagonal and its Frobenius sum
  `(n - |S|) + Σ_{i∈S} Aᵢ²`, the floor squared times `n`.

### 2. A re-supplied source at a random depth: one step, or a gain the loss cannot see

One direction of a re-injected linear loop: entry `x₀`, pass `h ↦ λ h + b`.

* `roll_add_sub`. `h_{T+k} - h_T = ((λ-1)x₀ + b) · λᵀ · Σ_{t<k} λᵗ`.
* `depth_invariant_iff`. For `λ ≥ 0` and `1 ≤ T₁ < T₂`, the exits agree iff `λ = 0` or the
  entry is the fixed point.
* `random_depth_zero_loss`. A depth-independent target read at a random depth with two
  depths in its support is fit exactly only by those two maps.
* `fixed_entry_flat`, `fp_zero_fixed_entry`. With `b = (1-λ)x₀` the state is `x₀` at every
  depth for every `λ`. The loss is the same for every gain, and so is the fixed-point term
  (it is zero).
* `roll_budget`, `budget_needs_expansion`. With `0 ≤ λ ≤ 1` and `|b| ≤ β`,
  `|h_T| ≤ |x₀| + βT`. A small entry and a far target force expansion. This is the plain
  loop's pass 0 (state x6.2, gain 1.25).

### 3. New evidence every pass: the loss asks for gain exactly 1

Evidence `x_t` (coordinate `t`) arrives at pass `t`; `h_{t+1} = λ h_t + b x_t` from 0.

* `evState_eq`, `evLoss_eq`. The squared error of reading the state as the sum of the
  arrived evidence is `Σ_{s<T} (b λˢ - 1)²`.
* `newEv_zero_iff`. For `T ≥ 2` the only zero is `b = 1, λ = 1`.
* `newEv_antitone`, `newEv_strict`, `newEv_monotone_above`, `newEv_random_antitone`. At
  `b = 1` the loss strictly falls as `λ` rises to 1, rises above 1, and the same holds for
  any law of depths.
* `relay_zero_iff`. If the reader wants only the newest evidence, the only zero is
  `b = 1, λ = 0`. Moving information needs no memory of the last pass, so it asks for no
  gain.

So the gain the loss asks for, direction by direction, is set by one property: must that
direction keep what earlier passes wrote? Accumulation: 1. Relay: 0. Re-supplied source:
0 or indifferent.

### 4. On a settled trajectory the recurrence's gradient is a write's gradient

Linear pass `h ↦ J h + ε G h + c`, fixed point `J h* + c = h*`, entry `h₀`.

* `hasDerivAt_blockRoll`. `d/dε` of the exit state at `ε = 0` is the first variation
  `v_{t+1} = J v_t + G a_t` (a real `HasDerivAt`, not a definition).
* `blockRoll_zero_eq`. The unperturbed rollout is `h* + Jᵗ (h₀ - h*)`.
* `firstVar_split`. `v_T = writeVar_T(G h*) + recVar_T(h₀ - h*)`: the variation of adding
  `G h*` to the per-pass write, plus a recurrence-only term driven by the transient.
* `write_rollout`. `writeVar` is exactly what adding `ε G h*` to the write does.
* `recVar_le`. `‖recVar_T‖ ≤ ‖h₀ - h*‖ · V_T`, `V` depending only on `‖J‖, ‖G‖, T`.
* **`settled_block_is_write`.** At `h₀ = h*` the derivative along `G` equals the
  derivative along the write `G h*`, exactly.

What this says for MORPH, by description, not by proof: any first-order benefit of changing
the core's recurrence along a settled slot trajectory is a benefit a change of the per-pass
write could deliver too, and the prelude already trains that write. The recurrence has a
gradient of its own only in proportion to how far the trajectory moves. The linear model
does not cover the nonlinear core; `recVar_le` is the quantitative form I would expect to
survive linearisation around the trajectory.

## Numerical checks (CPU, 2 cores, no GPU)

### Linear models (`linear_checks.py`, Poisson(6) depth clamped to [1, 8])

* New evidence, write `b` optimised: the random-depth loss has its minimum at `λ = 1.000`.
  The normalised fixed-point term averaged over depths falls monotonically on [0, 1] (1.983
  at 0, 0.213 at 1). Adding it moves the minimum to 1.002 (weight 0.1) and 1.015 (weight 1.0).
  This holds only for evidence that keeps arriving at every depth. When the work ends after a
  few passes (the toy below, and any real finite job) the term demands that the state stop
  moving afterwards, and the trained toy shows it pulling the map DOWN. I do not use the
  linear result for MORPH.
* Re-supplied source, entry at the target: all 100 gains in [0, 0.99] reach zero loss, with
  and without the fixed-point term. Entry off the target (`m = 1.5`): only `λ ≤ 0.01` reaches
  zero without the term, and the minimum sits at `λ ≈ 0.15` with it.

### A trained toy loop (`toy_map.py`)

A small copy of the core step: `h₀ = input_norm(prelude(x))`; each pass is the diagonal
injection on 20 of 64 dims (the same 31 % as MORPH, `A` init 0.447, `dt` init 1) then one
shared pre-norm attention + MLP block; Poisson(6) depth clamped to [1, 8], full BPTT, the
fixed-point term at weight 1.0, AdamW. Eight positions, each holding its index code, a
pointer and a 4-dim value. Tasks: `own` (its own value), `hopK` (the value `K` pointer hops
away), `sumhopK` (the sum of the values over hops 1..K), `ownhopK` (own + hop K, with a
bypass: the head also reads the prelude state). Map instrument: `core_map_fd.py`'s
definition at forced depth 6. Cotangent gain: `‖∂L/∂h_t‖ / ‖∂L/∂h_{t+1}‖`.

Two seeds per arm, 4000 steps (first group) or 6000 steps (second group), 512 fresh
eval rows. "Floor" uses the toy's own learned `A`. "Depth value" is MSE at depth 1 minus
depth 6 (target variance 1 to 5). Gains are the RMS over positions at each pass. Seed 0 / seed 1.

| arm (job) | FP | floor | typical gain pass 0 → 5 | positions > 1, pass 1 | cotangent gain pass 0 → 5 | depth value |
|---|---|---|---|---|---|---|
| `own`, bypass (none) | 1.0 | 0.865 | 0.834 flat / 0.838 flat | 0 / 0 | 0.99 → 0.54 / 1.00 → 0.54 | 0.0000 / 0.0001 |
| `own`, bypass (none) | 0.0 | 0.868 | 0.860 flat / 0.860 flat | 0 / 0 | 0.99 → 0.73 / 1.03 → 0.67 | 0.0001 / 0.0001 |
| `hop2` (relay, 2 passes) | 1.0 | 0.848 / 0.849 | 0.768 → 0.761 / 0.789 → 0.787 | 0 / 0 | 0.95 → 0.97 / 0.98 → 0.40 | 0.741 / 0.814 |
| `hop2` (relay, 2 passes) | 0.0 | 0.855 / 0.859 | 1.114 → 0.864 / 1.034 → 0.862 | 44 % / 30 % | 1.50 → 0.84 / 1.48 → 0.76 | 0.867 / 0.936 |
| `sumhop3` (accumulate, 3 passes) | 1.0 | 0.846 / 0.849 | 0.873 → 0.805 / 0.877 → 0.814 | 0.9 % / 1.1 % | 1.06 → 0.92 / 1.08 → 0.96 | 1.758 / 1.766 |
| `sumhop3` (accumulate, 3 passes) | 0.0 | 0.847 / 0.853 | 1.271 → 0.858 / 1.224 → 0.867 | 52 % / 61 % | 2.10 → 0.82 / 3.19 → 1.04 | 1.719 / 1.808 |
| `ownhop2`, bypass (relay) | 1.0 | 0.848 / 0.851 | 0.787 flat / 0.781 → 0.776 | 0 / 0 | 0.99 → 0.42 / 0.94 → 0.84 | 0.796 / 0.815 |

First group (prelude and coda each do one attention lookup, so `hop3` and below need only
one pass; the settled arms of the table above in all but name): every arm with a depth value
under 0.002 reads a flat map 0.79 to 0.84 with the term on and 0.863 to 0.876 with it off,
zero positions above 1. `hop4` seed 1 learned a two-pass job (depth value 0.008) and read a
flat 0.79.

What the toy says, stated only as far as it goes:

* **A settled loop reads its floor.** With no multi-pass job the map is flat across
  passes at the floor (term off) or 0.03 under it (term on). No position ever reads above 1.
  This is the slot loop's signature: flat on every pass including pass 0.
* **A loop that works across passes AND is let to keep moving reads like the plain loop.**
  Term off, relay or accumulation: pass 0 expands (1.03 to 1.27), 30 to 61 % of positions read
  above 1 on pass 1, and the gain falls pass by pass to the floor by pass 5. That is the plain
  loop's shape (1.25 on pass 0, 21 % above 1 on pass 1 falling to 3.4 %, 0.97 falling to 0.92).
* **The fixed-point term squeezes a working loop's map into a flat contraction without
  removing its depth value.** Term on, the same jobs read flat 0.76 to 0.88, almost nothing
  above 1, and still earn 0.74 to 1.77 of MSE from depth. So in the toy a flat contraction is
  NOT evidence of no work, and the map's level does not decide the depth value.
* **The injection decay moves when a task pushes on it.** The toy's `A` fell from 0.447 to
  0.30 to 0.34 on every multi-pass job (the re-supplied source, theorem 2: gain 0), and stayed
  at 0.44 to 0.46 on the settled ones. MORPH's `A` has not moved in any model, which says no
  MORPH loss pushes on it (the toy's lr is 10x MORPH's, so this is not a like-for-like
  comparison).
* **The loss-direction (cotangent) gain separates the regimes better than the typical gain.**
  Settled: it decays to 0.54 by the last pass. Accumulation with the term on: it stays at
  0.92 to 1.09 on every pass although the typical gain is 0.81. The typical gain averages over
  every direction and cannot see a demand that lives in a few.

## The ranked causes

1. **The slot trajectory has no work to keep (it is settled from the entry).** Theorems 2,
   3 and 4. Measured: slot norm ratio 1.02 then 1.006, map at the floor; plain state x6.2 on
   pass 0. Why it is settled: the coda's own token path and one attention pass over earlier
   slots already give the coda what the slot can carry (lessons 5 and 6 of the history doc:
   Conditions A and B). This is the root.
2. **The architecture supplies the number.** The 0.865 floor is `DiagonalInjection`'s init
   decay on 31 % of the channels, and `A` never moves. Reading "0.88" as a contraction the
   loss selected is a category error. It is the default a quiet core shows.
3. **The entry's distance from the operating point.** A far entry forces a transient
   (`roll_budget`), and the transient is exactly what gives the recurrence its own gradient
   (`recVar_le`). This is part of cause 1, and it is the mechanism behind the old reading that
   the noise entry made the plain loop earn more than the prelude entry (0.185 vs 0.033,
   lesson 8).
4. **The fixed-point term is a modulator, not the cause of the slot's 0.88.** On a settled
   loop it is zero along the settled family (theorem 2) and the toy moves only from 0.83 to
   0.86 when it is removed. On a WORKING loop it is strong: the toy's plain-like map (expansion,
   tail above 1, falling gain) becomes a flat contraction under it, with the depth value kept.
   So the term can produce a flat map on its own, and a flat map alone does not prove "no
   work". What says the slot loop has no work is not its map; it is the coda K-curve
   (0.001 to 0.002), the pass-0 reading (the slot's pass 0 is as flat as its pass 5; a working
   toy under the term still has its highest gain on pass 0), and the state's motion (norm
   ratio 1.02 then 1.006). The plain loop carries the same term at 1.0 and keeps its working
   shape, so in MORPH the term is weak against a real job.
5. **Not causes.** The row hinge does not bind after step 1000. Position diversity is a
   consequence of cause 1 per position, not a separate cause.

What I could NOT explain: the older E14 arm pressed on its 0.9 hinge (targets 0.95 and 0.98
gave 0.913 and 0.929, target 1.02 gave 0.99). Under this account that arm had upward pressure,
so something in it kept work across passes, or its `A` moved. Its checkpoint is gone, so I
cannot read its `A`.

## Discriminating interventions (on causes, not on the gain)

Instruments: `core_map_fd.py` fp32 at depth 6; the injection floor from `log_A` (CPU, one
minute); the coda K-curve and paired coda CE. I add one instrument the account needs: the
per-pass cotangent gain `‖∂L/∂h_t‖ / ‖∂L/∂h_{t+1}‖` along the loss's own direction, because
the typical gain averages over 4096 directions and cannot see a demand that lives in a few
(the e4 code direction moved its exit separation x2 while the typical gain did not move).

**I-1. Change the floor, change nothing else (tests cause 2 against "the loss selects
0.88").** `model.injection_channels: all` with `injection_all_decay: 0.9` on the e4probe
recipe (an existing knob, no code). Floor at init: `sqrt((704 · 0.81 + 320 · 0.2) / 1024) =
0.787`. With it the entry is already the fixed point of the non-context channels (dt = 0.1).
* Account true: the slot typical gain reads 0.77 to 0.81 (the new floor within 0.02), p10 to
  p99 spread under 0.05, zero readings above 1, coda K1−K6 ≤ 0.003, paired coda CE within
  ±0.01 of e4probe.
* Account false (the objective holds the map at about 0.88): the blocks compensate and the
  gain returns to 0.85 to 0.90.

**I-2. Take the per-pass source away after pass 0 (tests cause 1: give the slot something
to keep).** Build: a `_tul_core` flag that skips `self.injection(h, e)` for passes t ≥ 1 on
the slot loop (SCSE's `source_free` already does this for its own path). Now the ctx
channels lose the entry at `A` per pass unless the loss holds it.
* Account true: `log_A` rises (mean `A` ≥ 0.7 by step 5000, readable on CPU), the floor and
  the typical gain rise with it to about 0.93 or more, uniformly; the cotangent gain along
  the loss direction rises toward 1; coda K1−K6 stays ≤ 0.003 or goes negative, because
  keeping the entry is memory, not computation. That is: a map near 1 with no earning, made by
  a cause and not by a hinge.
* Account false: `A` stays at 0.44 and the map stays at 0.87 although the entry now washes
  out (`washout`, T5).

**I-0. Zero-training reading, today.** Run `core_map_fd.py` (and the cotangent gain) on
`slot-spandec-strict-prev-reach1` @5000. There the loop is the only relay, content `h` spans
back arrives at pass `h-1`, and a slot that wants several spans must keep what arrived
earlier (accumulation, theorem 3). Floor 0.864. Account: later passes read above the floor by
more than the ruler's +0.013 (I predict ≥ +0.03) and the gain rises with slot index in the
row (more spans behind); or, if the typical gain stays at +0.013, the cotangent gain along the
loss direction is near 1 while the ruler's decays. If both read like the ruler, the account
is wrong about what the slot loop's map reflects.

## What the two queued arms can and cannot tell

**Arm A, fixed-point term off.** This arm is more useful than its prereg says, because the
toy shows two distinct outcomes. (a) No hidden work (my account): the map stays flat across
passes at the floor or up to 0.03 above it (0.86 to 0.90), no position above 1, pass 0 no
higher than pass 5; the state moves more on later passes (toy: motion 0.09 to 0.11 at pass 5
vs 0.003); coda K1−K6 ≤ 0.003. A-1 (≥ 0.92) fails. (b) Work the term was squeezing: the
map takes the plain shape, pass 0 and pass 1 above the later passes, a tail above 1 on the
early passes, the gain falling pass by pass, and a coda K-curve appears. Outcome (b) would
refute my root cause for the slot loop. What arm A cannot tell: whether giving the loop
something to keep would raise the gain. Removing a settling pressure does not create work.

**Arm B, floor hinge at 0.95.** The account predicts B-1 holds, and that the cheapest way to
meet the hinge is the one parameter the loss never used: the injection decay. Floor 0.95
needs mean `A ≈ 0.83`; 0.98 needs `A ≈ 0.97`. Prediction: B's mean `A` at step 5000 ≥ 0.7,
and its measured typical gain within 0.03 of its new floor; spread under 0.05; coda K1−K6
≤ 0.003 (B-3 fails). A raised `A` also slows the ctx channels' settling (`Aᵀ`), which can put
a small slope on the K-curve with no value in it (lesson 2): read it only with paired CE.
What it can tell: whether a map near 1 is sufficient for depth use (the account says no). What
it cannot tell: the cause, because forcing the effect goes around it. Read `log_A` from B's
checkpoint before reading anything else. If `A` did not move and the blocks carried the
gain, that is the more interesting result and it would weaken cause 2's "default" reading.

## What is proved, checked, and conjectured

* Proved (Lean, no `sorry`): the floor bounds, the random-depth fixed-point
  characterisation and its flat family, the accumulation and relay optima, the budget bound,
  and the settled-trajectory derivative identity with its transient bound. All in linear or
  Lipschitz models.
* Measured (CPU, checkpoints): `A`, `dt`, the floors, the core weight RMS.
* Checked numerically (toys, two seeds): the fixed-point term's direction in the linear
  models; the trained toy's maps against its floor, with and without the term, on settled,
  relay and accumulation jobs.
* Conjectured: that MORPH's blocks are quiet on slot states BECAUSE the trajectory is
  settled (the theorem is linear; the brief's numbers and the floor fit it); that the plain
  loop's shape comes from work across passes that its fixed-point term is too weak to
  squeeze; every prediction in the intervention list.
* Found against my first draft: the toy refutes "the fixed-point term never lowers the map".
  It lowers a working loop's map a lot. The account survives only because the slot loop shows
  no sign of work by three readings that do not depend on the map.
