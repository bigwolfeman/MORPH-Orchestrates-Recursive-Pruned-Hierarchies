# Planned: the staged arm in the geometry where the loop is the only cross-span path

Status: failure

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

## Results

Filed 2026-09-10 19:30. `slot-mnext-staged-mask` (commit `7804793`; `tul_slot_mnext_staged` plus
`tg_restrict: true` at scope "all", `use_kernels: false`, `tg_scoped_kernels: true`): HEALTHY
to 4,999, tripwire max 151 at step 1689 (ruler 78.7, threshold 1e4), 9,349 tok/s at step 200
(floor 8,086; the fully eager E4 twin read 9,168), smoke peak 11.18 GB, wall clock 51 min 07 s
(ruler 52 min 38 s, 0.97x). Runner final val_loss 4.5062 (ruler 4.3775; unmasked staged 4.2988).

Sweeps, 480 rows, forced depths 1/2/3/6/9/12/16 (`mux_local` is the training objective, see
the staged filing; the forecast column is `mux_local_next`, the memory column `mux_local_own`):

| step | tokens K1−K6 | tokens K3−K6 | next K1−K6 | next @1 / @3 / @6 (ruler @6; staged @6) | own @1 / @3 / @6 | CE@6 (ruler; staged) |
| --- | --- | --- | --- | --- | --- | --- |
| 2,500 | +0.0024 [+0.0021, +0.0028] | +0.0020 | +0.0582 | 6.9590 / 8.6649 / 6.9008 (6.8766; 6.8815) | 6.1301 / 3.5319 / 6.4315 | 4.8183 (4.7194; 4.7638) |
| 5,000 | +0.0033 [+0.0028, +0.0037] | +0.0022 [+0.0018, +0.0026] | +0.0604 [+0.0564, +0.0647] | 6.8540 / 8.6382 / 6.7936 (6.7724; 6.7698) | 5.9294 / 3.2388 / 6.4151 | 4.4373 (4.3290; 4.2483) |

Every earlier mask arm on this family read token K1−K6 between 0.020 and 0.060 (E4
`to-mnext-y2-mask` 0.0204; E13 `m12-mnext-mask` 0.033 / 0.049; E18 `mask-k2/k4/k8` 0.024 to
0.049; all under the absmean ternary rule). This one, under `norm_match` with the staged loss,
reads 0.0033: the mask's forced token dependence is gone. The forecast curve is the unmasked
staged arm's curve (+0.060 vs +0.067) with the same shape: own-span CE 3.24 at pass 3, next-span
8.64 at pass 3 → 6.79 at pass 6, and an exit 0.021 nats WORSE than the ruler's depth-6 forecast.
The mask price on the tokens is the usual one: 0.108 nats behind the ruler and 0.189 behind
the unmasked staged arm at 5k.

Worth profile at 5,000 (offsets 0..6 after the slot): zero +0.336 [+0.314, +0.359], +0.291,
+0.162, +0.149, +0.113, +0.081, +0.059 (unmasked staged +0.078 .. +0.021; ruler +0.094 ..
+0.006); shuffle +0.447, +0.171, +0.121, +0.100, +0.070, +0.046, +0.022; wrong_seed +0.177,
+0.059, +0.047, +0.047, +0.031, +0.022, +0.020. The slot state is load-bearing here (4x the
unmasked arm at every offset), which is what the mask is for. State probe: |h| 187 at depth 1
→ 211 → 220 → 230 at 6 → 238 at 16 (unmasked staged 449 → 496 → 511); relative distance from
depth 1 0.163 / 0.249 / 0.313 / 0.444, cos 0.996 / 0.986 / 0.981 / 0.951.

Gradient probe at 5,000 (12 rows, batch 2, depth 6, eager; self-check max rel err 2.3e-3,
flagged pass): combined cancellation ratio **0.385** (ruler 0.520, unmasked staged 0.442,
every-pass 0.771); MUX alone 0.404, token CE alone 0.410. Per-pass share of the shared core
weight gradient 0.121 / 0.264 / **0.478** / 0.023 / 0.049 / 0.065; per-pass cosine to the
total +0.29 / **−0.39** / +0.90 / +0.31 / **−0.49** / +0.58 (two negative passes; the unmasked
staged arm had one, pass 6 at −0.22). Cotangent share 0.221 / 0.228 / 0.267 / 0.091 / 0.089 /
0.104. Parameter-group norms: prelude 31.1, core.residual 20.6, coda 3.26, core.attention
1.82, core.mlp 1.80, W_prefix 0.49 (the mask routes far more gradient through the prelude and
the HC residual than the unmasked arm: 6.49 / 10.7 / 1.78 there). z-optimisation probe (12
rows, batch 2, 200 Adam steps): ce_loop 4.2265, entry **+0.0235** (ruler +0.0015; unmasked
staged +0.0073; every arm before: +0.0002 to +0.0033), zero +0.1012, shuffle +0.0767, fitted z
−1.663 (random start −0.853; cos(z*, z_loop) +0.954), loop z rank 13.0.

Probe caveat, on the record: the two probes were meant to start after the next arm's rate
check (`slot-mnext-staged-fullread`), behind a free-memory guard of 11 GB. The guard matched
the stale `SLOTLOOP3 COMPLETE` line from 17:00 and started them at 19:21, one minute into the
fullread trainer's compile, with 13.6 GB free. Nothing OOMed (fullread passed its rate check
at 19:22:45 at 11,861 tok/s) and both probes exited 0, but the ordering was luck. Artifacts:
`lab/experiments/results/2026-09-10-slot-mnext-staged-mask/`; npz under
`ignored/experiment-artifacts/2026-09-10-slot-mnext-staged-mask/`; probe JSON under
`ignored/experiment-artifacts/2026-09-10-slot-gradient-probe/` and `-slot-z-optimize/`.

## Verdict

P-a TRUE. **P-b FALSE** (0.0033 against a 0.03 bar given 72 %; the highest-confidence
prediction in the file). P-c TRUE by the bar (+0.060 > 0.02, K3−K6 +1.84) with the exit 0.021
nats worse than the ruler's. **P-d TRUE** (+0.0235 > 0.02): the first arm on which the coda
reads the loop's exit as better than the loop's entry by more than noise. P-e TRUE (passes 2
and 5 negative). P-f TRUE (0.385 < 0.520). P-g FALSE as predicted (0.108 behind). P-h TRUE
(0.97x). Binding clause 1 (P-c TRUE and P-g FALSE) fires on its letter and fails on its
reading: the clause assumed the forecast curve moving under the mask meant "the loop earns
only when it is the only path", and the token curve says the loop does NOT earn depth even
as the only path (0.0033). What the mask changed is the READER'S use of the exit (+0.0235
vs +0.0073 unmasked) and the size of the passes' disagreement (two negative passes,
cancellation 0.385), not the depth the exit depends on. The mask's own historic token
K-curve (0.020 to 0.060 on six arms) did not reproduce under `norm_match` with the staged
loss; no plain-mask arm exists under `norm_match`, so whether the ternary rule or the loss
removed it is one arm away (`tul_slot_mux_norm_match` + `tg_restrict`).

## Updated hypothesis

Under the mask the coda uses the loop's exit (worth +0.34 nats at the first token after the
slot, +0.0235 over the entry), and the exit is still what ONE pass makes: the own→next
conversion the staged loss imposes on passes 4–6 runs at the same size with and without the
mask, and the token curve is flat both ways. The geometry decides how much the reader needs
the slot; it does not decide whether the loop's later passes contribute. Two readings now
point the same way: the loop's later passes do work only when a loss gives them a job the
first pass has not already finished, and the coda's targets (next tokens) are a job one pass
finishes. The queued `slot-mnext-staged-all` (a different job at every pass) and
`slot-mnext-staged-fullread` (a reader that sees the whole carrier) are the last two
loss/reader arms; `slot-mnext-parcae-core` and `slot-mnext-gradpass` change the map and the
input. If all four leave the exit at one pass of the ruler, the next question is the target,
not the loop: what a slot could be asked for that needs more than one pass.
