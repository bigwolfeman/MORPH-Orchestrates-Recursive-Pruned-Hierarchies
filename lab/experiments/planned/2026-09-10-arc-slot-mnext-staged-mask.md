# Planned: the staged arm in the geometry where the loop is the only cross-span path

Status: planned

Date: 2026-09-10 (frozen before launch). Arc: `2026-09-04-loop-contribution-arc.md`.
One factor against the staged arm [`slot-mnext-staged`](2026-09-10-arc-slot-mnext-staged.md),
which is on the GPU as this file is written. Source of the mechanism: question 4 of
[`lab/toy_slot_loop/WRITEUP.md`](../../toy_slot_loop/WRITEUP.md). Design note:
[`2026-09-10-credit-assignment-in-the-slot-loop.md`](../../../.agents/notes/proposed/architecture/2026-09-10-credit-assignment-in-the-slot-loop.md).

## Question

The toy study has two findings and only one of them is about the loss. The second one is
about GEOMETRY, and it is the larger of the two. Under the toy's `strict` geometry — the
prelude attends within a span, the core attends the previous slot, a token reads only the
previous span's prefix cells — the slot loop is the ONLY cross-span path, depth has an exact
price, and the winning loss attachment solved the chain. Under `permissive`, which is
MORPH's own geometry (a prelude causal over everything, a coda reading a causal chain of
prefix cells), ONE core pass reached 0.283 nats and 66 % accuracy on the same task, and the
loop's own contribution fell from +0.514 to +0.106 K1−K6 with the attachment held fixed.
The toy's own words: in the shipped geometry the loop is one of at least three cross-span
paths, and it is the slowest to learn.

MORPH has that strict geometry in the tree: `tul.tg_restrict`, at the default scope "all",
masks the prelude AND the coda to same-span-or-slot, so a token reaches an earlier span only
through that span's slot positions. It is also the ONE slot arm that has ever read a token
K1−K6 above 0.02 on this tree (E16, 0.406 on clean math). Does the staged attachment earn
depth when it is the only path?

## Hypothesis

H-mask-1: staging gives the intermediate passes a reachable job, and the mask gives the exit
state a job nothing else can do. The two are complementary, and the toy predicts the largest
effect exactly here — its winning cell was measured in the strict geometry. So the token
K-curve and the forecast K-curve both move, at a CE cost the mask always charges.

H-mask-0: the mask has run four times on this tree (E4, E13, E16, E17) and every time it
bought a large token K-curve together with a WORSE model — 0.18 nats behind the plain model
paired at depth 12 on math, 0.155 behind on Sudoku, flat over T from 2 to 16. On that
reading the K-curve the mask produces is a measure of forced dependence, not of value
(exactly what E6 established for the deep draw: K-diffs measure depth DEPENDENCE, not depth
value), and the staged attachment changes nothing about that.

Both readings predict a large token K-curve. They disagree about whether the model is any
good, which is why the CE-at-matched-depth comparison is in the binding rule and not only in
the predictions.

## Method

`tul_slot_mnext_staged_mask` = `tul_slot_mnext_staged` (`mux_stage_own_iters: 3`, itself the
ruler `slot-mux-norm-match` plus that knob) plus `tul.tg_restrict: true` at the default
`tg_restrict_scope: all`, with the kernel settings every mask arm since E16 carries.

Scope. "all" applies the same-span-or-slot mask in the PRELUDE and in the CODA, which is the
geometry the toy's finding is about: neither can compose across spans by itself. Scope
"coda" (the `slot-unpack-*` family) deliberately leaves the prelude global and is the wrong
control here.

Kernels, and why they are not a free choice. `tg_restrict` forces `model.use_kernels: false`
at construction (the fused window/CSA/HCA kernels do not know about the restriction and a
silent unmasked kernel path is forbidden), and that turns every fused kernel off
process-wide. `model.tg_scoped_kernels: true` is legal with scope "all" — every TG-restricted
branch stays eager BY CONSTRUCTION, because a prelude/coda window call always carries
`tg_allow` and routes to the reference path and the TG compressed branches are pure eager
functions — and it lets HC-Cayley, the CCA prologue and the core-region window run fused.
Two reasons it is required:

* **Rate.** E4's fully-eager mask arm on this exact recipe (`to-mnext-y2-mask`, mean-6, batch
  6, seq 1024) read **9,168 tok/s** at step 200, against the ruler's 12,429 and the rate
  floor of 8,086 — 13 % of margin. The mean-12 mask arms, fully eager, read 5,473
  (`m12-mnext-mask`) and 6,734 (`k12-mnext-mask`), both of which would be stopped by that
  rule today. E16 measured `tg_scoped_kernels` at 1.63x and −5.4 GB, so this arm is expected
  near the ruler's rate; the honest prediction is that it clears the floor either way, and
  the flag is what makes that comfortable rather than marginal.
* **The hinge.** The eager finite difference reads gain 0.94 ± 0.014 on a map whose true gain
  is 0.87 (noise bias at `slot_gain_eps` 0.02), so an eager arm's hinge at target 0.9 fires
  on noise while the ruler's, running fused, does not. Without the flag the arm would differ
  from its partner by the mask AND by whether the stability constraint is active.
  (`.agents/notes/proposed/bug-fix/2026-09-08-slot-gain-eps-noise-bias.md`.)

Side effect, named because it makes this arm a partial control for audit finding F1: under
`tg_restrict` the pooled `GatedPoolCompressor` is not built on ANY layer
(`_CCAHCAAttention.__init__` sets `self.compressor = None`), so the HCA compressed branch
that reads exactly 0.000 at S = 64 on the ruler cannot exist here. This arm therefore changes
the geometry AND removes F1; `slot-mux-hca-fix` is the arm that removes F1 alone.

Everything else is the staged arm: M-next MUX at beta 1 through the tied head, the own-span
term at pass 3, prelude entry, hinge lambda 100 at 0.9, `core_fixed_point_lambda` 1.0,
`slot_cot_clip` 4.0, `ternary_scale_mode: norm_match`, seq 1024, batch 6, seed 1, 5,000
steps, ramp 1,000.

Runner `arc/run_slotloop3.sh`, a 12-step smoke first, the draw under the sustained tripwire.
Rate rule: tok/s below 8,086 at step 200 skips the arm and the queue continues.

Readout: `core_depth_sweep.py` at 2,500 and 5,000, depths 1,2,3,6,9,12,16 — the token
K-curve plus `mux_local_next_final` and `mux_local_own_final`; `worth_profile.py` and
`slot_state_probe.py` at 5,000; `slot_gradient_probe.py` and `slot_z_optimize.py` at 5,000 by
hand. No new wandb keys.

The numbers this arm is read against, cited once. The one-factor partner
(`slot-mnext-staged`) had not finished when these predictions were frozen, so the bars are
set against the RULER `slot-mux-norm-match`: 480-row CE at depth 6 = **4.3290** at 5,000;
token K1−K6 +0.0001, K3−K6 −0.0000; `mux_local` K1−K6 +0.0067, K3−K6 +0.0005; wall clock
52 min 38 s; cancellation ratio 0.520; per-pass cotangent 0.168 / 0.168 / 0.166 / 0.160 /
0.154 / 0.183; z-optimisation `ce_loop` 4.1173 with the loop ENTRY worth +0.0015. The mask's
own prior on this tree: token K1−K6 0.406 (E16, clean math) at 0.18 nats behind the plain
model; 0.155 behind on Sudoku; every rating bucket flat from T = 2 to 16.

## Predictions (frozen)

- **P-a (survival AND rate).** HEALTHY to 5,000 with no sustained tripwire AND tok/s at step
  200 at or above 8,086: **78 %**. The mask has never detonated on this recipe and E4's arm
  cleared the floor fully eager at 9,168; the 22 % is mostly the rate rule at a recipe that
  now also carries the staged term's extra readout, plus the usual detonation base rate.
- **P-b (tokens).** Token K1−K6 at 5,000 above 0.03: **72 %**. Above 0.10: **50 %**. This is
  the ONE arm family on this tree that has ever produced a large token K-curve, and the
  reason is structural: the mask makes the slot the only route, so the tokens have no
  alternative to reading it. High probability, and deliberately NOT read as success — see
  the binding rule.
- **P-c (forecast K-curve, the bar that decides the toy's claim).** `mux_local_next` K1−K6
  above 0.02 at 5,000 (the ruler: 0.0067): **40 %**. K3−K6 above 0.002 (the ruler: +0.0005):
  **28 %**. Highest of the four arms in this batch because it combines the toy's winning
  attachment with the toy's strict geometry, which is the only cell that solved its chain.
  Under 50 % because E16's mask arm read `answer` K3−K6 at −0.030 — WORSE past pass 3 — and
  nothing in the tree says staging repairs that.
- **P-d (the reader).** The coda's exit-minus-entry CE from `slot_z_optimize.py` above 0.02
  nats (the ruler: +0.0015): **55 %**. The highest of the batch and close to a mechanical
  consequence: with the coda masked, z carries information no other path supplies, so
  replacing it with the entry state should cost real nats. If this reads low on a mask arm
  the probe itself is suspect.
- **P-e (a negative pass).** At least one pass reads a NEGATIVE cosine to the total
  core-weight gradient in `slot_gradient_probe.py`: **30 %**. Same mechanism as the staged
  arm, in a geometry that gives the exit's job more to do; the toy saw the separation at the
  LAST pass.
- **P-f (cancellation).** Combined cancellation ratio BELOW the ruler's 0.520: **35 %**. The
  token CE now flows through the slot, which changes what the token side asks of the loop
  for the first time in this batch; direction unknown, and the toy's correlation is a
  correlation.
- **P-g (val CE, and the cost).** 480-row CE at depth 6 within 0.05 of the ruler's 4.3290:
  **15 %**. The mask has cost 0.1-0.2 nats on every panel that ran it, and there is no reason
  for this one to be different. Recorded as the PRICE, not as a verdict.
- **P-h (cost).** Wall clock within 1.2x of the ruler's 52 min 38 s: **60 %**. E4's fully
  eager arm ran at 0.74x the ruler's rate; `tg_scoped_kernels` bought 1.63x on E16's arm, so
  the expected landing is at or slightly above the ruler's rate, but that 1.63x was measured
  at seq 512 / batch 24 on a different draw and does not transfer by right.

## Binding

- P-c TRUE and P-g FALSE (the forecast curve moves AND the model pays the usual mask price)
  ⇒ the toy's geometry finding transfers and the loop earns only when it is the only path.
  That is a statement about MORPH's architecture, not about the loss: the next question is
  whether a cheaper restriction (scope "coda", or the E18 width axis) buys the same
  dependence at less CE, and the horizon goes to 20k.
- P-b TRUE with P-c FALSE ⇒ the mask bought token dependence without the loop earning
  anything on its own target, which is E16's result reproduced under the staged attachment.
  The mask's K-curve is then confirmed as forced dependence, and the loss-attachment lane is
  closed for good.
- P-c FALSE and P-d FALSE ⇒ even as the only cross-span path the loop does not deliver
  anything the reader uses. That would be the strongest negative available on this
  architecture and it points at the state, not the training: ~50 slot cells at effective rank
  1.7-4.8 in 1024 dimensions.
- A rate stop ⇒ the arm is skipped, the queue continues, and the next mask attempt has to
  buy its speed back before it is worth GPU time.

## Not verified before launch

The arm has not run on a GPU: CPU build and a config compose check (`tg_restrict` ON, scope
"all", `use_kernels` False, `tg_scoped_kernels` True, `mux_stage_own_iters` 3). No test in
this change exercises the mask itself — `tg_restrict` has its own suite
(`tests/test_tg_restrict.py`) and this arm adds no code. The RATE is the biggest unverified
number here: 9,168 tok/s is E4's fully-eager arm at the same recipe and 1.63x is E16's
scoped-kernel speedup measured at a different sequence length, batch and depth draw, so the
predicted landing rate is an extrapolation from two measurements, not a measurement. The
interaction of `tg_scoped_kernels` with the STAGED term (an extra `[B, S, V]` readout on an
intermediate state) is unmeasured. The claim that `tg_restrict` removes F1 by construction is
read from `_CCAHCAAttention.__init__`, not measured on this arm's checkpoint. The one-factor
partner had not finished when this file was frozen.
