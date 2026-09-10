# Planned: the M-next slot loop under Bansal's progressive loss

Status: failure

Date: 2026-09-10 (frozen before launch; Wolfe's framing after the per-pass gradient probe:
"the loop issue is credit assignment"). Arc: `2026-09-04-loop-contribution-arc.md`.
Follows `results/2026-09-10-slot-gradient-probe/README.md` and the two flat forward-lever
panels (`failures/2026-09-10-arc-slot-map-levers.md`,
`failures/2026-09-10-arc-slot-reread.md`). Sibling arm on the same question:
[`2026-09-10-arc-slot-mnext-per-pass-lora.md`](2026-09-10-arc-slot-mnext-per-pass-lora.md).
Design note:
[`2026-09-10-credit-assignment-in-the-slot-loop.md`](../../../.agents/notes/proposed/architecture/2026-09-10-credit-assignment-in-the-slot-loop.md).

## Question

The gradient probe closed the "the gradient does not reach the passes" explanation: under
the prelude entry the cotangent arriving at each of the six passes is FLAT (share 0.168 /
0.168 / 0.166 / 0.160 / 0.154 / 0.183 on `slot-mux-norm-match`), and the MUX pays for the
loop 7.3x over the token CE (3.5e-3 vs 4.8e-4 into the loop state). What it opened is
CANCELLATION: `|sum_t dW_t| / sum_t |dW_t|` on the shared core weights is 0.520 combined
(0.601 token CE, 0.550 MUX), against 0.41 for six orthogonal equal-norm vectors and 1.0
for six aligned ones, with per-pass cosines to the total from 0.27 to 0.75. The six passes
of one weight-shared map are asking for near-orthogonal updates.

One reading of that: each pass is being trained on a state distribution only IT ever sees,
because full BPTT lets the map co-adapt with its own trajectory — pass 3 only ever has to
handle what passes 1 and 2 produced today. Bansal, Schwarzschild et al. 2022 attack exactly
this with the progressive loss: a random number of passes runs with NO gradient, the rest
with gradient, so the map must improve ANY state it is handed and cannot settle at the
identity or lean on the pass index. Does that change what the slot loop's passes agree on,
and does anything downstream (the token K-curve, the MUX K-curve) read it?

## Hypothesis

H-prog-1 (for): the cancellation is co-adaptation. Cutting a random prefix out of the
gradient forces every pass to be a useful map on states it did not itself produce, the
per-pass updates align, and the loop starts earning depth — first on the MUX forecast
(the target that already pays for the loop) and, if the coda reads it at all, on tokens.

H-prog-0 (against, the null this tree keeps confirming): the cancellation is a symptom, not
the disease. The core blocks move the slot state by 2-20 % per pass against 86-110 % on
token states (`failures/2026-09-10-arc-slot-map-levers.md`), so the slot loop's map is
near-inert whatever its gradient looks like; a training-side change to WHICH passes carry
gradient cannot make a near-identity map earn depth, and the arm reads flat like every
lever before it.

## Method

`tul_slot_mnext_progressive` = `tul_slot_mux_norm_match` (the ruler `slot-mux-norm-match`)
plus ONE change: `tul.progressive_p: 0.5`.

The mechanism, as implemented: inside `_tul_core`, with probability 0.5 a slot whose
realised Poisson depth `T_i` is at least 2 draws `k_i` uniform in `[1, T_i - 1]` and its
first `k_i` passes have their OUTPUT detached, so no cotangent enters at those positions
and they contribute no weight gradient through that slot; the remaining slots run full BPTT
exactly as the ruler does. The loss is unchanged (token CE + MUX at beta 1). `k_i <= T_i-1`
keeps every slot's LAST pass in the gradient window, so the terminal fixed-point term and
the exit state always sit on a grad pass. The gain hinge measures only GRAD passes (its
penalty shapes the core weights, and a detached position is one where the objective was cut
away). `progressive_p: 0.0` is bit-identical to the pre-change tree — verified directly on
the tiny CPU model, loss `9.1006078720092773` and sha256 `32b174ef…2853f8` over all 208
gradient tensors at master `c429e22` and after the change.

Everything else is the ruler: M-next MUX at beta 1 through the tied head, prelude entry,
hinge lambda 100 at target 0.9, `core_fixed_point_lambda` 1.0, `slot_cot_clip` 4.0,
`ternary_scale_mode: norm_match`, seq 1024, batch 6, seed 1, 5,000 steps, ramp 1,000,
`tg_scoped_kernels`.

Runner `arc/run_slotloop3.sh` (file-driven queue, commit pinned in `arc/slotloop3_arms.txt`),
a 12-step smoke first, the draw under the sustained tripwire
(`lab/divergence/tripwire_sustained.py`). Rate rule: tok/s below 8,086 at step 200 skips
the arm and the queue continues (the ruler read 12,429).

Readout: `core_depth_sweep.py` at checkpoints 2,500 and 5,000, depths 1,2,3,6,9,12,16,
reporting both the token K-curve and the `mux_local` forecast K-curve on 480 rows;
`worth_profile.py` and `slot_state_probe.py` at 5,000 (the runner's `slot` kind does both);
`slot_anatomy.py` and `slot_gradient_probe.py` at 5,000 by hand — the gradient probe is the
arm's own instrument here, since the cancellation ratio is the quantity the mechanism
targets. New wandb keys to read alongside: `loop/prog_frac`, `loop/prog_nograd_passes`,
`loop/prog_grad_depth`.

The ruler's numbers, cited once: 480-row CE at depth 6 = **4.3290** at step 5,000
(`sweep_5000.log`; the trainer's own final val_loss was 4.3775 and the last `[VAL]` line
kept in `run.log` is step 4,750 at 4.4486 — there is no `[VAL 5000]` line in that log);
token K1-K6 +0.0001 [-0.0000, +0.0002], K3-K6 -0.0000; `mux_local` K1-K6 +0.0067
[+0.0053, +0.0081], K3-K6 +0.0005; wall clock 52 min 38 s for 5,000 steps
(21:41:42 -> 22:34:20 in `arc/queue.log`); combined cancellation ratio 0.520.

## Predictions (frozen)

- **P-a (survival).** HEALTHY to 5,000, no sustained tripwire (`preclip/total > 1e4` at
  step >= 200): **80 %**. Reasoning: the ruler ran HEALTHY with a max preclip of 78.7 at
  step 245, and this arm strictly REMOVES gradient paths rather than adding any; the
  detonation modes on this tree are backward products through the loop and a first-pass
  scale runaway, and a shorter grad tail can only shrink the first. The 20 % is for the
  hinge, which now measures a smaller, differently-distributed slot set each step.
- **P-b (tokens above the plain ruler).** Token K1-K6 at 5,000 above 0.03 (the plain
  prelude-entry ruler `plain-panel-norm-match`'s value): **12 %**. Above 0.10 (the plain
  NOISE-entry value from E19's `parcae-entry`): **4 %**. Reasoning: every slot arm run
  under norm_match on any entry, coda or target has read token K1-K6 at or under 0.005,
  and the reason is structural — the coda reads the slot state ONCE, and the levers panel
  measured that state as barely moved by the core blocks at any pass. A change to which
  passes carry gradient does not add anything for the coda to read at depth 6 that was not
  there at depth 1. I do not expect this to move.
- **P-c (forecast K-curve).** `mux_local` K1-K6 above 0.02 at 5,000: **30 %**. Reasoning:
  the MUX is the one loss that already pays for the loop (7.3x the token CE into the loop
  state) and the one K-curve that is not exactly zero on the ruler (0.0067). If the
  progressive cut does anything at all, the first place it must show is here — a map
  trained to improve arbitrary states should read better at 6 passes than at 1 on the
  target it is actually supervised on. Three-fold headroom over 0.0067 is a real bar and
  I put it under even money. K3-K6 above 0.002 specifically: **18 %**.
- **P-d (cancellation).** Combined cancellation ratio from `slot_gradient_probe.py` at
  5,000 above 0.60 (the ruler: 0.520): **35 %**. Above 0.70: **15 %**. Reasoning: this is
  the quantity the mechanism targets and the most direct test of H-prog-1, but the probe
  measures the passes at a FIXED depth 6 with the prefix mechanism off, so it reads the
  trained map's agreement, not the training-time cut. A mechanism that works should still
  leave the trained map's passes more aligned. Against it: the probe's own reading is that
  pass 6 carries 36 % of the norm and the early passes 10-15 %, and the progressive cut
  does not change that imbalance — it only changes which passes see gradient on a given
  step.
- **P-e (val CE).** 480-row CE at depth 6 within 0.05 of the ruler's 4.3290: **55 %**.
  Reasoning: half the slots lose gradient on a random prefix each step, which is a real
  reduction in the gradient signal per step, and 5,000 steps is a short horizon where that
  usually costs a little. I expect a small loss rather than a gain, and I am explicitly
  NOT treating a CE gap at 5,000 steps as a verdict on the mechanism (Wolfe 2026-09-09:
  short-horizon CE cannot rank looped against unlooped).
- **P-f (cost).** Wall clock within 1.2x of the ruler's 52 min 38 s: **90 %**. Reasoning:
  the mechanism adds two `torch.rand` draws of shape [B, S] and one `torch.where` per
  pass; the prefix passes still run in the forward, and their backward gets cheaper, not
  more expensive.

## Binding

- P-c TRUE (`mux_local` K1-K6 > 0.02) and P-d TRUE at 0.60 => credit assignment is the
  lever: the next arm combines the progressive cut with whatever the per-pass LoRA arm
  says, and a 20k horizon read follows before any further lever search.
- P-d TRUE but P-b and P-c FALSE => the mechanism does what it claims to the gradient and
  nothing downstream reads it. That localises the failure past the backward and onto the
  map's near-inertness (the levers panel's structural finding), and the next arm has to
  change what a pass CAN do to the state, not how it is trained.
- All flat => the progressive loss joins the list of levers that read flat on this loop.
  Combined with the per-pass LoRA arm, both credit-assignment attacks would then have
  failed, and the honest next step is not another arm on this loop: it is the question of
  whether ~50 slot cells carrying a rank-2-to-5 state can support a depth-6 map at all.

## Not verified before launch

Neither arm has run on the GPU: everything above is a CPU build, a CPU test suite and a
config compose check. `torch.compile` behaviour of the new per-slot mask path is untested
until the runner's 12-step smoke — the mask is a plain `torch.where` on a `[B, S]` bool, so
there is no data-dependent Python branch, but that is an argument, not a measurement. The
hinge under progressive prefixes is tested only for its MASK (a unit test proves it never
probes a detached position); its interaction with the constraint over 5,000 steps — whether
a smaller, resampled slot population per step makes `loss/gain_est` noisier or the penalty
weaker — is unmeasured. `progressive_p: 0.5` is Bansal's own default shape, not a value
tuned on this tree; no sweep of p is planned before this draw. The counters
`loop/prog_*` have been read on the tiny CPU model only.

## Results

Filed 2026-09-10 12:50. `slot-mnext-progressive` (commit `dc37b40`, `tul.progressive_p 0.5`):
HEALTHY to 4,999, tripwire max 616 at step 245 (ruler 78.7, threshold 1e4), 13,108 tok/s at
step 200, smoke peak 10.85 GB, wall clock 47.6 min against the ruler's 52.6 min (0.90x).
Trainer `[VAL 4750]` 4.4142 (ruler 4.4486); runner final val_loss 4.3462 (ruler 4.3775).

Sweeps, 480 rows, forced depths 1/2/3/6/9/12/16:

| step | tokens K1−K6 | tokens K3−K6 | `mux_local` K1−K6 | `mux_local` K3−K6 | CE@6 (ruler) |
| --- | --- | --- | --- | --- | --- |
| 2,500 | +0.0004 [+0.0002, +0.0005] | −0.0000 | +0.0062 [+0.0047, +0.0076] | −0.0004 | 4.6988 |
| 5,000 | +0.0006 [+0.0005, +0.0008] | +0.0001 [+0.0000, +0.0002] | +0.0088 [+0.0076, +0.0101] | +0.0004 [−0.0002, +0.0010] | 4.2938 (4.3290) |

Worth profile at 5,000 (offsets 0..4): zero +0.037 [+0.030, +0.044], +0.051, +0.024, +0.020,
+0.015; shuffle +0.031, +0.044, +0.027, +0.021, +0.016. State probe at 5,000: |h| 96 at depth 1
→ 117 at 3 → 125 at 6 → 133 at 16; relative distance from the depth-1 state 0.28 / 0.42 /
0.59, cos 0.99 / 0.97 / 0.92; per-pass movement does NOT decay (0.09 at 3, 0.15 at 6, 0.24
at 16): the map keeps moving the state past the trained depth, which is what a
progressive loss trains for, and the token CE does not change with it. Gradient probe at 5,000
(`slot_gradient_probe.py`, 12 rows, depth 6, self-check 1.8e-7): combined cancellation
ratio |sum_t dW_t| / sum_t |dW_t| **0.497** (ruler 0.520); per-pass share of the norm sum
0.07 / 0.08 / 0.09 / 0.10 / 0.20 / 0.46 (ruler 0.15 / 0.10 / 0.11 / 0.12 / 0.17 / 0.36);
per-pass cosine to the total 0.15 / 0.19 / 0.44 / 0.57 / 0.22 / 0.72. The MUX still pays
for the loop (|total| 1.13 vs token CE 0.15). z-optimisation probe (`slot_z_optimize.py`,
same rows): ce_loop 4.0839, entry state +0.0057, zero +0.0169, shuffle +0.0129, fitted z
−0.887 (random start −0.795), loop z rank 13.7. Artifacts:
`lab/experiments/results/2026-09-10-slot-mnext-progressive/`; npz in
`ignored/experiment-artifacts/2026-09-10-slot-mnext-progressive/`.

## Verdict

P-a TRUE. P-b FALSE on both bars (0.0006). P-c FALSE on both clauses (0.0088 < 0.02; K3−K6
0.0004 within its CI of 0.002 but under it). P-d FALSE (0.497, below the ruler's 0.520 and both bars). P-e TRUE (0.036
better). P-f TRUE (0.90x). H-prog-1 (the progressive cut makes the passes agree and the
loop earn) is not supported on the readable clauses; the binding clause "all flat" holds. The mechanism did what it says to the forward: the state keeps moving past
depth 6 where the ruler's stops, so the map learned to act on states it did not build, and
the coda's CE is indifferent to that motion.

## Updated hypothesis

Same as the noise-entry filing: the coda's CE with the loop's exit equals its CE with the
loop's entry (z-optimisation probe, 0.0002–0.006 nats on every arm read, this one 0.0057), while a fitted z
is worth 0.9–2.6 nats. A training rule on the passes cannot change what the reader asks
for; the state the loop writes is not in the directions the coda uses, and the MUX target
(next-span tokens through the tied head) is met in one pass from the prelude state. The
question that remains, per the binding, is whether any slot target NEEDS iteration; the
per-pass LoRA arm is the last parameterisation lever queued.
