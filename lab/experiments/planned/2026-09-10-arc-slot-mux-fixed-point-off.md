# Planned: the terminal fixed-point term OFF, which is the only honest way to test it here

Status: planned

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
