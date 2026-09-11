# Planned: no MUX, masked. The slot's target becomes whatever the coda needs

Status: planned

Date: 2026-09-10 (frozen before launch). Arc: `2026-09-04-loop-contribution-arc.md`.
One factor against `slot-loop-norm-match`
([`failures/2026-09-09-arc-slot-loop-norm-match.md`](../failures/2026-09-09-arc-slot-loop-norm-match.md),
`morph/configs/tul_slot_loop_norm_match.yaml`, `mux_beta` absent so 0) and one factor
against the masked MUX arm
[`slot-mux-mask-norm-match`](2026-09-10-arc-slot-mux-mask-norm-match.md), queued beside it.
Design note:
[`2026-09-10-credit-assignment-in-the-slot-loop.md`](../../../.agents/notes/proposed/architecture/2026-09-10-credit-assignment-in-the-slot-loop.md).
Geometry claim: question 4 of
[`lab/toy_slot_loop/WRITEUP.md`](../../toy_slot_loop/WRITEUP.md).

## Question

Every arm in this batch and the four before it changed WHERE the slot's target is applied,
or HOW the loop is trained toward it. None of them changed the fact that somebody chose the
target. On every MUX arm the slot is supervised toward the next span's tokens through the
tied head, which is a guess about what a slot should hold, frozen into the loss at beta 1.
Wolfe's reading after today's numbers: the slot's TARGET is the lever, and it should be
defined by what the coda needs at the tokens.

There is exactly one configuration in this tree where that is literally true. Turn the MUX
off, so the coda's token cross-entropy is the only loss that touches the slot state. Turn
`tg_restrict` on, so the prelude and the coda cannot compose across spans and the loop is
the only route from an earlier span to a later token. The slot's target is then not
specified at all. It is discovered, and it is exactly the part of the token loss that no
other path can carry.

The known objection is the whole reason this arm needs the mask. The gradient probe
measured that without the MUX the shared core receives about 1 % of the prelude's gradient
norm, and that the MUX pays for the loop 7.3x over the token CE (3.5e-3 against 4.8e-4 into
the loop state). The no-MUX slot loop is the emptiest arm on this tree: token K1-K6
-0.0000, the coda loses 0.0086 nats when the slot state is zeroed, and 0.0002 nats when the
loop's six passes are replaced by their own entry state. The mask is the one change that
can raise the token CE's own stake in the loop, because it deletes the alternatives.
Whether it raises it enough is the arm.

## Hypothesis

H-target: the MUX has been the problem, not the solution. A forecast target that one pass
can satisfy gives the later passes nothing to do, and it dominates the gradient 7.3x, so the
token side never gets a say in what the slot holds. Remove it, force the token loss through
the slot, and the loop is trained on a genuinely unsatisfied objective for the first time.
Prediction: token K1-K6 rises above the no-MUX arm's -0.0000 and above the masked MUX arm's
value, and the coda's exit-minus-entry CE rises well above +0.0002.

H-starve: the MUX was carrying the loop, and the mask cannot replace it. The token CE's
cotangent at the loop state is 4.7e-4 with or without the mask; what the mask changes is the
ROUTE, not the magnitude. With the MUX off the core gets a weak and now also a narrower
signal, and the arm reads flat at a worse CE than either parent. Prediction: token K1-K6
below 0.005 and a CE at depth 6 worse than the masked MUX arm's.

The pair discriminates. Both readings predict a CE price; they disagree about whether
anything is bought with it.

## Method

`tul_slot_loop_mask_norm_match` = `tul_slot_loop_norm_match` plus `tul.tg_restrict: true`
at the default `tg_restrict_scope: all`, with `model.use_kernels: false` (forced by the
restriction at construction) and `model.tg_scoped_kernels: true`. Nothing else changes: no
MUX (`mux_beta` absent, so 0), prelude entry, hinge lambda 100 at target 0.9,
`core_fixed_point_lambda` 1.0, `slot_cot_clip` 4.0, `ternary_scale_mode: norm_match`,
`tul.max_slots` 64, `activate_at` 0.0, seq 1024, batch 6, seed 1, 5,000 steps, ramp 1,000,
checkpoints at 2,500 and 5,000.

Scope, kernels and the F1 side effect are identical to the masked MUX arm and are argued
once, there. In short: scope "all" masks the prelude and the coda; `tg_restrict` forces the
fused kernels off and `tg_scoped_kernels` buys most of them back on the branches the
restriction does not touch; under `tg_restrict` the pooled `GatedPoolCompressor` is not
built, so this arm also removes audit finding F1.

Buildability, checked rather than assumed. A no-MUX slot loop under `tg_restrict` has never
been built in this tree, `tul_setup.py` raises on unknown `tul.*` keys, and several knob
combinations refuse at construction. It builds: a tiny CPU model with `tg_restrict=True`,
`mux_beta=0.0`, `tokens_through_core=False`, the gain constraint and the fixed-point term
ran a training forward and a backward (loss 4.728560) whose gradient reaches the core
(sum of absolute core gradients 9.96e+01). The core is not disconnected from the loss when
the MUX is off.

What is NOT available on this arm, so the readout is not silently narrower than it looks.
There is no `mux_local` column: the forecast K-curve, which is the bar in every other file
in this batch, does not exist here. The sweep's token K-curve is the whole depth reading,
and `worth_profile.py` plus `slot_z_optimize.py` carry the rest. That is a real loss of
instrument and it is the price of the question.

Runner `arc/run_slotloop3.sh` (file-driven queue, commit pinned in `arc/slotloop3_arms.txt`),
a 12-step smoke first, the draw under the sustained tripwire. Rate rule: tok/s below 8,086
at step 200 skips the arm and the queue continues.

Readout: `core_depth_sweep.py` at 2,500 and 5,000, depths 1,2,3,6,9,12,16;
`worth_profile.py` and `slot_state_probe.py` at 5,000 (the runner's `slot` kind);
`slot_gradient_probe.py` and `slot_z_optimize.py` at 5,000 by hand. No new wandb keys.

The numbers this arm is read against, cited once. `slot-loop-norm-match` at 5,000 (the
one-factor parent): token K1-K6 **-0.0000** [-0.0001, +0.0000], K3-K6 -0.0000; CE at depth
6 minus the plain model's, token-paired, **+0.164** [+0.156, +0.173]; worth zero 0.051;
runner val 4.2601; 12,011 tok/s; `gain_est` mean 0.868 with the hinge binding 0.1 % of
steps. Its probes at 5,000: token CE 4.3994, fixed-point term 0.00023, per-pass cotangent
share 0.159 / 0.159 / 0.159 / 0.159 / 0.166 / 0.199, per-pass share of the core weight
gradient 0.045 / 0.045 / 0.051 / 0.073 / 0.368 / 0.419, per-pass cosine to the total
0.607 / 0.680 / 0.691 / 0.631 / 0.311 / 0.217, combined cancellation ratio **0.344**;
z-optimisation token CE 4.0222, zeroing the slot state costs 0.0086, shuffling 0.0192,
replacing the exit with the loop's ENTRY costs **+0.0002**, and a gradient-fitted z is worth
at least 1.185. The RULER `slot-mux-norm-match` at 5,000, for the CE scale: 480-row CE at
depth 6 4.3290, token K1-K6 +0.0001, `mux_local` K1-K6 +0.0067, z-opt entry +0.0015,
cancellation 0.520, 12,429 tok/s, wall clock 52 min 38 s, rate floor 8,086. The masked MUX
arm of today, `slot-mnext-staged-mask`: token K1-K6 +0.0033, CE at depth 6 4.4373, worth
zero +0.336, z-opt entry +0.0235, 9,349 tok/s.

## Predictions (frozen)

- **P-a (survival AND rate).** HEALTHY to 5,000 with no sustained tripwire AND tok/s at step
  200 at or above 8,086: **82 %**. The no-MUX arm is the calmest arm in the panel
  (`gain_est` 0.868, the hinge binding 0.1 % of steps) and today's masked arm read 9,349
  tok/s with a MUX readout this one does not pay for. The 18 % covers the detonation base
  rate and the one genuinely new thing here, which is a core trained almost entirely by a
  narrow token loss.
- **P-b (the bar: tokens).** Token K1-K6 at 5,000 above 0.010: **35 %**. Above 0.020:
  **22 %**. Above 0.003, the masked MUX lineage's current value: **50 %**. The mask is the
  only lever that has ever moved this number, and this is the configuration where the token
  loss has the largest possible stake in the loop. It sits well under even money because the
  token CE's cotangent at the loop state did not change between arms with and without the
  MUX (4.7e-4 both ways), and because the mask's own K-curve may already be an `absmean`
  artefact, which the arm queued beside this one is built to decide.
- **P-c (the reader, and the sharpest number here).** The coda's exit-minus-entry CE from
  `slot_z_optimize.py` above 0.02 nats (this arm's parent: **+0.0002**, the smallest in the
  record; today's masked arm +0.0235): **50 %**. A 100x move on the parent. Even money
  because the mechanism is strong (nothing else can carry cross-span information) and the
  starting point is the weakest of any arm.
- **P-d (worth).** `worth_profile.py` zero at offset 0 above 0.20 nats (the parent 0.051;
  today's masked arm +0.336; E4 0.213): **70 %**. Two mask arms under two ternary rules have
  read above 0.20, and this one removes every alternative route.
- **P-e (cancellation).** Combined cancellation ratio BELOW the parent's 0.344: **30 %**.
  The parent is ALREADY the lowest in the record, close to the 0.408 of six orthogonal
  equal-norm vectors and below it, so there is little room. Recorded because the toy's
  correlation between low cancellation and earning is the reason anyone would look, and
  because a value far below 0.344 with a flat K-curve would be direct evidence against
  reading cancellation as a sign of learning.
- **P-f (where the gradient sits).** The per-pass share of the core weight gradient stays
  back-loaded, with passes 5 and 6 together above 0.60 (the parent: 0.368 + 0.419 = 0.787):
  **65 %**. Without a MUX the only loss is at the exit, so the back-loading is structural;
  the mask changes the route, not the attachment point. A FLAT profile here would be the
  surprise and would say the mask changed what the early passes are for.
- **P-g (val CE, the price).** 480-row CE at depth 6 within 0.05 of the masked MUX arm's
  4.4373: **30 %**. Two prices stack here, the mask's 0.1 to 0.2 nats and the missing MUX,
  and the parent already sits +0.164 behind the plain model token-paired. Recorded as the
  price, not as a verdict.
- **P-h (cost).** Wall clock within 1.2x of the ruler's 52 min 38 s: **80 %**. The mask's
  cost with scoped kernels was 0.97x on today's arm, and dropping the MUX readout removes a
  `[B, S, V]` term rather than adding one.

## Binding

- P-b TRUE at the 0.010 bar AND P-c TRUE ⇒ the target was the lever. A loop trained only by
  what the coda cannot get anywhere else earns depth and delivers something the reader uses.
  That reframes every flat arm in this arc as a consequence of the MUX's chosen target, and
  the next run is this arm at 20k with a matched-compute shallow control, not another
  attachment.
- P-b FALSE and P-c TRUE ⇒ the reader needs the slot and the loop's DEPTH is still not what
  it needs. Same reading as today's masked arm, reached from the opposite side (no chosen
  target at all), which would make it the strongest statement in the arc that the exit state
  is what one pass produces regardless of what trains it.
- P-b FALSE and P-c FALSE ⇒ H-starve. The token CE cannot train this loop even when it is
  the only path, and the MUX was load-bearing. The lane after that is not the loss and not
  the geometry: it is the state, which sits at effective rank 12 to 13 in 1024 dimensions on
  every arm measured.
- P-d TRUE with P-b FALSE ⇒ the slot is load-bearing and depth-free, which is the mask's
  standing signature and not new information. It would mean the mask buys dependence on the
  SLOT, never on the loop.
- A rate stop ⇒ the arm is skipped and the queue continues.

## Not verified before launch

The arm has not run on a GPU. What was run: a Hydra compose check printing
`tul.tg_restrict` True at scope "all", `tul.mux_beta` absent (so 0 at setup),
`tul.tokens_through_core` False, `model.use_kernels` False, `model.tg_scoped_kernels` True,
`model.core_fixed_point_lambda` 1.0, `ternary_scale_mode` norm_match, 5,000 steps at batch 6
and seq 1024; and a tiny CPU build of the same knob combination that ran a training forward
and a backward reaching the core (loss 4.728560, core gradient sum 9.96e+01). That build is
a WIRING check at d_model 64 with 2 core blocks. It says the combination is legal and the
core is connected; it says nothing about whether the token loss can train the loop at the
real shape. No test in this change exercises the mask, which has its own suite
(`tests/test_tg_restrict.py`); this arm adds no code. The rate is an extrapolation from
today's masked MUX arm (9,349 tok/s) with the MUX removed, not a measurement. Memory at the
real shape is unmeasured. Whether `slot_gradient_probe.py` and `slot_z_optimize.py` behave
identically on a no-MUX arm under `tg_restrict` is untested here, although both ran on the
unmasked no-MUX arm and the probe's MUX terms are guarded by a `has_mux` branch. The
missing `mux_local` column means this arm cannot be compared to the rest of the batch on
the forecast bar at all.
