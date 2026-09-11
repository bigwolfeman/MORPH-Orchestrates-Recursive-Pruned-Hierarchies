# Planned: the terminal fixed-point term OFF, which is the only honest way to test it here

Status: failure

Date: 2026-09-10 (frozen before launch). Arc: `2026-09-04-loop-contribution-arc.md`.
One factor against the ruler `slot-mux-norm-match`
(`morph/configs/tul_slot_mux_norm_match.yaml`): `model.core_fixed_point_lambda` 1.0 to 0.0.
Source of the mechanism: question 3 of
[`lab/toy_slot_loop/WRITEUP.md`](../../toy_slot_loop/WRITEUP.md). Shipping record for the
term:
[`2026-09-07-fixed-point-objective-as-the-loop-stability-term.md`](../../../.agents/notes/implemented/architecture/2026-09-07-fixed-point-objective-as-the-loop-stability-term.md).

## Question

The toy study ranked its levers and the terminal fixed-point penalty came first. On a chain
task that needs iteration, `lambda * ||z - h_{T-1}||^2 / ||z||^2` at the slot's last pass
took the escape rate from 2 of 5 seeds to **5 of 5**, took the value CE from 0.7693 to
0.0000, took K1-K6 from +0.514 to +1.282, and cost 233 s against 198 s. Nothing else in the
toy came close: per-pass LoRA changed nothing, the injection settings moved one seed, and
the noise entry was the strongest poison.

The obvious transfer is to raise the term on the real model. That is the wrong test, and
saying why is most of this file.

First, the ruler already runs it. `base.yaml` sets `model.core_fixed_point_lambda: 1.0` and
nothing in the ruler's 17-file defaults chain touches the key; a compose check prints 1.0.
In the slot loop the term lives in `_tul_core` (shipped 7ff72a0, 2026-09-07) on every valid
slot at its LAST iteration, grad iterations only, training only. Every slot arm in the
record since 2026-09-07 has carried it, and none of them has ever run without it.

Second, the term is already satisfied. On the ruler's own `probe.jsonl` the live
`loss/fixed_point` averages **0.00798** over steps 4,900 to 4,999, down from 0.0220 at steps
200 to 299; the gradient probe's 12-row recompute at a forced depth of 6 reads 0.00302 (the
no-MUX arm 0.00023). Either way the term contributes under a hundredth of a nat to a
combined loss of 11.28, and it is FALLING on its own. The loop also already contracts: the
ruler's `loop/delta_ratio_t*` over the same steps reads 0.818 / 0.202 / 0.117 / 0.088 /
0.076 / 0.070 / 0.067 / 0.065, so after the first pass sets the scale each later pass moves
the state by 7 to 12 % of its norm and the exit pass by 6.5 % (`slot-mnext-staged` reads
0.042 / 0.038 / 0.035 at t5 to t7). Raising lambda pushes harder on a constraint the map
already meets. It would change the loss by a number too small to see and would teach us
nothing about the toy's finding.

The honest one-factor test of a term in that state is whether REMOVING it changes anything.
That is this arm.

## Hypothesis

H-holds-it-flat: the term is a contraction penalty, and a contraction penalty rewards a map
whose last pass does as little as possible. On the toy that was the ingredient that made the
chain solvable, because the toy's map had real work to do and the term stopped it thrashing.
On MORPH the loop is already near-inert (per-pass movement 9.3 / 4.3 / 2.7 / 2.0 / 1.7 / 1.6
/ 1.6 % on the ruler's anatomy), so the same term may be paying the map to stay there. Under
this reading lambda 0 lets the later passes move, and the forecast K-curve and the per-pass
movement profile both open up, at some risk to stability.

H-inert: at 0.003 the term is numerically negligible against a loss of 11.28, its gradient
is a rounding error beside the MUX's, and removing it reproduces the ruler inside its
confidence intervals. The toy's strongest lever then does not transfer, and the reason is
the state of the map it acts on: the toy's loop had `||z - h_{T-1}||` worth constraining,
MORPH's does not.

Both readings share a stability tail. On the PLAIN model the term is the second measured
hold against the detonation (0 of 6 warmup-0 detonations against 4 of 7 controls). This arm
keeps the FIRST and stronger hold, the 1,000-step LR ramp (0 detonations in 9 of 9 draws),
and keeps the slot-loop gain constraint. E13 and E14 measured the term as free on the
healthy slot-loop arms and as no hold at all against the two slot-loop failure modes it was
not built for. So the bet is survivable, not free, and the tripwire scores it.

## Method

`tul_slot_mux_fixed_point_off` = `tul_slot_mux_norm_match` with
`model.core_fixed_point_lambda: 0.0`. Nothing else changes: M-next MUX at beta 1 through
the tied head, exit-only attachment, prelude entry, hinge lambda 100 at target 0.9 with
`slot_gain_eps` 0.02, `slot_cot_clip` 4.0, `ternary_scale_mode: norm_match`, fused kernels
on, seq 1024, batch 6, seed 1, 5,000 steps, ramp 1,000, checkpoints at 2,500 and 5,000.

What the knob actually does, read from the code rather than from the config. In `_tul_core`
the term is collected only while `_fp_lam > 0.0` and the pass carries gradient. At lambda 0
the collection does not happen, `self._core_aux` is not set, and the forward's output dict
has NO `fixed_point` or `fp_weighted` key. Both wandb log sites guard on key presence
(`morph/training/train.py`), so `train/fixed_point`, `train/fp_weighted` and
`loss/fixed_point` simply disappear from this run rather than logging zero.
`slot_gradient_probe.py` guards the same way and will omit the term from its header line.
Consequence for the readout, named so it is not discovered later: this arm cannot report the
quantity it removes. The comparison on that number is the ruler's 0.00302 against nothing,
and the arm's own instrument for the same mechanism is the per-pass movement profile from
`slot_state_probe.py` and `loop/delta_ratio_t*` in `probe.jsonl`.

Configs that already zero the key, listed so this arm is not read as new:
`notul_no_fixed_point`, `notul_depth1`, `notul_parcae_entry`, `tul_slot_unpack_free`,
`tul_slot_mnext_noise_entry`. None of them is the ruler with one factor changed: the first
three are plain-model arms, and the last two change the reader or the entry at the same
time.

Runner `arc/run_slotloop3.sh` (file-driven queue, commit pinned in `arc/slotloop3_arms.txt`),
a 12-step smoke first, the draw under the sustained tripwire
(`lab/divergence/tripwire_sustained.py`). Rate rule: tok/s below 8,086 at step 200 skips the
arm and the queue continues.

Readout: `core_depth_sweep.py` at 2,500 and 5,000, depths 1,2,3,6,9,12,16, giving the token
K-curve and the forecast column `mux_local`; `worth_profile.py` and `slot_state_probe.py` at
5,000 (the runner's `slot` kind does both); `slot_gradient_probe.py` and
`slot_z_optimize.py` at 5,000 by hand. `loop/delta_ratio_t*` and `loop/core_gain_t0` from
`probe.jsonl` are the arm's own instruments and neither hypothesis is scored on them.

The numbers this arm is read against, cited once. RULER `slot-mux-norm-match` at 5,000:
480-row CE at depth 6 **4.3290**; token K1-K6 +0.0001 [-0.0000, +0.0002], K3-K6 -0.0000;
`mux_local` K1-K6 +0.0067 [+0.0053, +0.0081], K3-K6 +0.0005; forecast at depth 1 / 6 =
6.7791 / 6.7724; worth zero at offset 0 +0.094; z-optimisation `ce_loop` 4.1173 with the
loop ENTRY worth +0.0015; combined cancellation ratio 0.520; per-pass cotangent share
0.168 / 0.168 / 0.166 / 0.160 / 0.154 / 0.183 and per-pass cosine to the total 0.33 / 0.27 /
0.37 / 0.63 / 0.75 / 0.67; fixed-point term **0.00798** live over steps 4,900 to 4,999 and
0.00302 on the gradient probe's 12 rows at a forced depth of 6; `loop/delta_ratio_t0..t7`
0.818 / 0.202 / 0.117 / 0.088 / 0.076 / 0.070 / 0.067 / **0.065**; `gain_est` mean 0.887
with the hinge binding 0.5 % of steps; 12,429 tok/s at step 200; wall clock 52 min 38 s;
tripwire max **78.7**; runner final val_loss 4.3775. Ruler anatomy, per-pass movement of the
full carrier: 65 / 17 / 9 / 6 / 4 / 4 / 3 %. Toy reference: escape 5/5 with the term against
2/5 without, K1-K6 +1.282 against +0.514.

## Predictions (frozen)

- **P-a (survival).** HEALTHY to 5,000 with no sustained tripwire (`preclip/total > 1e4` at
  step >= 200): **80 %**. The ramp is the measured hold and it is on; E13 and E14 read the
  term as free on healthy slot-loop arms, which cuts both ways and means removing it should
  also be free. The 20 % is larger than the batch's usual because this arm deliberately
  removes a stability term, and because the slot loop's live failure is the map's gain
  drifting to 1 at full BPTT under ternary, which is exactly what a terminal contraction
  penalty would otherwise damp.
- **P-b (tripwire size, a graded version of P-a).** Maximum `preclip/total` over the run
  above 3x the ruler's 78.7: **35 %**. Recorded separately because a run can survive and
  still show the term was doing something, and because the tripwire max is the cheapest
  reading of that.
- **P-c (the bar: forecast).** `mux_local` K1-K6 at 5,000 above 0.02 (the ruler +0.0067):
  **20 %**. K3-K6 above 0.002 (the ruler +0.0005): **15 %**. The lowest forecast bar in this
  batch. A term reading 0.003 has very little to give back when removed, and six arms have
  now failed to move this number from the training side. The 20 % is for H-holds-it-flat
  being right about direction even if the term is small.
- **P-d (tokens).** Token K1-K6 at 5,000 above 0.010: **10 %**. The token curve has not
  moved on any arm in this arc that left the reader and the geometry alone, and this arm
  changes neither.
- **P-e (the mechanism's own instrument, and the best-paired number in this file).**
  `loop/delta_ratio_last` from `probe.jsonl`, averaged over steps 4,900 to 4,999, above
  **0.090** (the ruler: 0.0650; `slot-mnext-staged`: 0.0348): **40 %**. This is the direct
  reading of whether the term was holding the exit pass still, it is measured on the same
  key in both runs, and it is the number to look at FIRST if P-c is false. Secondary and
  NOT scored: `slot_state_probe.py` relative distance from depth 1 to depth 6, because the
  ruler's own state probe was never run (its `absmean` twin read 0.346 and the staged arm
  0.215, which is too wide a band to set a bar in).
- **P-f (the reader).** The coda's exit-minus-entry CE from `slot_z_optimize.py` above 0.02
  nats (the ruler +0.0015): **12 %**. The reader's indifference has survived every arm that
  did not mask the coda; nothing here touches the reader.
- **P-g (cancellation).** Combined cancellation ratio BELOW the ruler's 0.520: **45 %**.
  Near even, and honestly so: removing a term that constrains only the LAST pass should
  change that pass's contribution and leave the other five alone, and the direction of the
  resulting agreement is not predictable from anything measured.
- **P-h (val CE).** 480-row CE at depth 6 within 0.05 of the ruler's 4.3290: **65 %**. The
  term contributes 0.003 nats to the training loss, so a large CE move would itself be
  evidence that the term does something the size of its value does not suggest. Recorded and
  NOT treated as a verdict (Wolfe 2026-09-09: short-horizon CE cannot rank looped against
  unlooped).
- **P-i (cost).** Wall clock within 1.1x of the ruler's 52 min 38 s: **90 %**. The arm
  removes two extra core applications' worth of bookkeeping per step and adds nothing. The
  toy measured the term at 233 s against 198 s, so if anything this arm should be slightly
  FASTER; a slower run would be a measurement problem, not a result.

## Binding

- P-c TRUE or P-e TRUE ⇒ the term was holding the exit still, and the toy's strongest lever
  transfers with its SIGN REVERSED on the real model: what helps a loop that has work to do
  suppresses a loop that does not. That is a statement about the whole stability-term
  family, because the gain hinge and `slot_cot_clip` are the same kind of object, and the
  next arm is those two at zero on the same ruler.
- P-c FALSE and P-e FALSE and P-a TRUE ⇒ the term is inert on this loop in both directions.
  The toy's ranking does not transfer, the reason is that MORPH's map has no fixed-point
  error worth 0.003 nats to constrain, and every future toy finding has to be checked
  against the size of the quantity it acts on BEFORE it earns GPU time. The term stays
  shipped on the strength of its plain-model detonation record, which this arm does not
  test.
- P-a FALSE (a detonation) ⇒ the term IS holding the slot loop together under `norm_match`
  at full BPTT, which no measurement has shown and E13/E14 argued against. That would be a
  new stability fact and it belongs in `lab/divergence/DIVERGENCE-README.md` the same day.
- P-b TRUE with P-a TRUE ⇒ a partial version of the above: the term damps the pre-clip
  gradient without being necessary. Worth recording in the divergence README as a graded
  reading, not as a hold.

## Not verified before launch

The arm has not run on a GPU. What was run: a Hydra compose check printing
`model.core_fixed_point_lambda` 0.0 with `tul.tg_restrict` False, `tul.mux_beta` 1.0 at
`mux_target` next, `model.use_kernels` True, `ternary_scale_mode` norm_match, 5,000 steps at
batch 6 and seq 1024, and the same check on the ruler printing 1.0; and a tiny CPU build at
lambda 0 that ran a training forward and a backward reaching the core (loss 9.060484, core
gradient sum 3.05e+02) and returned NO `fixed_point` key, against the same model at lambda
1.0 which returned 0.0611. That is the knob's effect verified on the forward, at d_model 64
with 2 core blocks, not at the real shape. The claim that the disappearing wandb keys are
harmless is read from the two guarded log sites in `morph/training/train.py` and the guard
in `slot_gradient_probe.py`; no run has been scored with those keys missing, so a downstream
reader that assumes their presence may still fail. The ruler's `loop/delta_ratio_t*` and
`loss/fixed_point` figures were read out of its `probe.jsonl` for this file and averaged
over the last 100 steps by hand; they are not in any earlier filing, and the arithmetic has
not been reviewed by anyone else. The stability claim rests on E13/E14 readings
taken under `absmean`, and no slot-loop arm has ever run without this term under
`norm_match`, which is the point of the arm and also the reason its survival probability is
the lowest in the batch.

## Results

Filed 2026-09-11 03:10. `slot-mux-fixed-point-off` (commit `69bab63`; the ruler with
`model.core_fixed_point_lambda` 0.0): survived to 4,999 with the runner verdict AMBIGUOUS:
maximum `preclip/total` **1,441 at step 245** (the ruler's maximum is 78.7 at the SAME step
245, so this is the ruler's largest gradient event 18x larger; threshold 1e4; no other step
above 236 after the first eight warmup steps, the ruler has none). 11,611 tok/s at step 200
(ruler 12,429), smoke peak 10.76 GB, training peak 11.87 GB, wall clock 49 min 07 s (ruler 52
min 38 s, 0.93x). Runner final val_loss 4.3697 (ruler 4.3775). At step 4,999: `gain_est`
0.896 (the hinge at target 0.9 sits at its edge; ruler 0.87), `loop/core_gain_t0` 5.5
(masked ruler-family 2.0), and the mechanism's own instrument, `loop/delta_ratio_last`
averaged over steps 4,900 to 4,999: **0.440** (ruler 0.065 on the same key and steps; 0.330
at steps 2,400 to 2,500).

Sweeps, 480 rows, forced depths 1/2/3/6/9/12/16:

| step | tokens K1−K6 | tokens K3−K6 | forecast K1−K6 | forecast K3−K6 | forecast @1 / @6 (ruler @6) | CE@6 (ruler) |
| --- | --- | --- | --- | --- | --- | --- |
| 2,500 | +0.0007 [+0.0005, +0.0008] | +0.0000 | +0.0109 [+0.0094, +0.0124] | −0.0008 | 6.8864 / 6.8755 (6.8766) | 4.7112 (4.7194) |
| 5,000 | +0.0005 [+0.0003, +0.0007] | −0.0000 [−0.0001, +0.0001] | **+0.0145 [+0.0127, +0.0166]** | +0.0000 [−0.0008, +0.0009] | 6.7928 / 6.7783 (6.7724) | 4.3144 (4.3290) |

The forecast K1−K6 doubles against the ruler (+0.0145 vs +0.0067) and stays under the 0.02
bar, with K3−K6 exactly zero: whatever the released exit pass does, the forecast has it by
pass 3. The tokens are flat. CE is 0.015 BETTER than the ruler at depth 6 and 0.008 at the
runner's val (a horizon reading, not a verdict).

Worth profile at 5,000 (offsets 0..6): zero +0.083 [+0.073, +0.092], +0.074, +0.035, +0.032,
+0.022, +0.018, +0.015 (ruler +0.094 .. +0.006); shuffle +0.056, +0.062, +0.041, +0.035,
+0.023, +0.015, +0.006; wrong_seed +0.032, +0.038, +0.020, +0.020, +0.016, +0.012, +0.010.
State probe, the reading that says what the term was doing: |h| **344 at depth 1 → 459 → 554
→ 767 at depth 6 → 1,308 at depth 16**; relative distance from depth 1 0.728 / 0.984 /
**1.667** / 3.217; cos 0.842 / 0.811 / 0.699 / 0.609; per-pass relative step 0.73 / 0.43 /
0.53 / 0.79 (the ruler-family arms with the term contract: masked ruler 0.152 / 0.232 /
0.350 / 0.516 with the norm 123 → 165; staged 0.215 at depth 6). Without the term the slot
state EXPANDS through the loop by a factor 2.2 over six passes and 3.8 over sixteen, and its
direction keeps turning. The fixed-point term was the thing bounding the trajectory.

Gradient probe at 5,000 (12 rows, batch 2, depth 6, eager; self-checks PASS at max rel err
1.5e-7): combined cancellation ratio **0.563** (ruler 0.520); MUX 0.558, token CE 0.550.
Per-pass share of the shared core weight gradient 0.287 / 0.190 / 0.108 / 0.112 / 0.090 /
0.214 (front-loaded; the ruler's pass-1 share was not the largest); per-pass cosine to the
total +0.70 / +0.63 / +0.37 / +0.57 / +0.26 / +0.54 (no negative pass). Cotangent share 0.230
/ 0.163 / 0.150 / 0.139 / 0.145 / 0.173, clip never binds. Parameter-group norms: prelude
2.81, coda 2.13, core.residual 0.84, core.mlp 0.82, core.attention 0.45. z-optimisation probe
(12 rows, batch 2, 200 Adam steps, bit-exact): ce_loop 4.0993, entry **+0.0084** (ruler
+0.0015; 5.6x, under the 0.02 bar), zero +0.0248, shuffle +0.0188; the fitted z reaches only
−0.153 (ruler −0.944) with cos(z*, z_loop) +0.989 and rank 6.2 (random start −0.221). Probe
caveat: the loop's z has norm 766 here against 152 on the ruler-family, and 200 Adam steps at
lr 0.01 cannot move a vector of that norm far, so the oracle bound is NOT comparable across
these two arms; the entry-vs-exit number is (both states are the model's own).

Probes ran 03:01 to 03:09 after the Parcae arm's rate check, 13.6 GB free at start.
Artifacts: `lab/experiments/results/2026-09-10-slot-mux-fixed-point-off/`; npz under
`ignored/experiment-artifacts/2026-09-10-slot-mux-fixed-point-off/`; probe JSON under
`ignored/experiment-artifacts/2026-09-10-slot-gradient-probe/` and `-slot-z-optimize/`.

## Verdict

P-a TRUE (survived; verdict AMBIGUOUS by the runner's graded rule, HEALTHY by the 1e4 abort
rule). **P-b TRUE** (1,441 > 236, given 35 %). P-c FALSE at both bars as predicted (+0.0145;
K3−K6 0.0000). P-d FALSE as predicted (+0.0005). **P-e TRUE** (0.440 > 0.090, given 40 %; the
number the file said to read first). P-f FALSE (+0.0084 against 0.02). P-g FALSE (0.563 is
above 0.520, given 45 % for below). P-h TRUE (0.015 better). P-i TRUE (0.93x). Six of nine
held, but the two that missed are the two that carry the mechanism, and they missed against
the file's stated lean (H-inert), so the file goes under failures. Binding clause 1 fires on
P-e and its reading has to be cut in half: the term WAS holding the exit still (delta ratio
0.065 → 0.440, the state's norm 2.2x over the loop instead of contracting, the step-245
gradient event 18x larger), and releasing it did NOT let the loop earn anything: forecast
K3−K6 zero, tokens flat, the reader +0.008. The "sign reversed" half of the clause is not
supported; the term is not inert and the loop's contribution is indifferent to it. Clause 4
also fires (P-b with P-a): the term damps the pre-clip gradient without being necessary,
recorded as a graded reading in `lab/divergence/DIVERGENCE-README.md`.

## Updated hypothesis

The fixed-point term is a state bound, not a contribution lever. With it the slot trajectory
contracts (relative motion 0.2 to 0.35 over six passes); without it the trajectory expands
2.2x and keeps turning, the hinge sits at its edge (0.896 against target 0.9), and the
largest gradient event grows 18x, while every readout of what the loop DELIVERS (tokens,
forecast past pass 3, worth, entry-vs-exit) stays where the ruler put it. The toy's ranking
transfers in the sense that the term changes the trajectory a lot; it does not transfer in
the sense that matters, because on the real model the trajectory's shape is not what the
readouts depend on. The term stays shipped on its plain-model detonation record, and the
slot-loop lane after this batch is the state's TARGET, not its dynamics: an expanding state
that reads the same as a contracting one is a state whose content nobody is asking for.
