# SCSE stall — evidence collected DURING the campaign (2026-08-25)

Kept here so it survives; folded into the experiment writeup at the end.

## Symptom
seed 1: control final val CE 4.6863, SCSE 6.4277 (+1.74 nats).
SCSE TRAINING loss is flat from step 200: 6.53 / 6.55 / 6.38 / 6.53 at 200 / 1000 / 3000 / 3400.
Control over the same span: 6.58 / 5.67 / 4.87 / 5.23. So SCSE STOPS LEARNING, it does not
merely learn worse. Val curves separate between step 250 and 500.

## Evidence 1 — the deviation explodes (live probe, batch 1, seed-1 checkpoints)
||Delta_t|| / ||h*|| across the loop:
  step 1000: 0.10 -> 1.98 -> 9.27 -> 17.45 -> 25.40 -> 32.67
  step 3000: 0.10 -> 8.30 -> 8.04 ->  8.48 ->  9.17 ->  9.84
Mask fully active (1.00) at every iteration, so this is not the frozen-loop failure.
Delta_0 enters at ~0.10x the anchor and jumps ~80x in ONE iteration.
h_T = h* + Delta_T is therefore ~90% Delta_T: the fixed reference SCSE exists to provide is
swamped by the very quantity it is supposed to anchor.

## Evidence 2 — the core is driven expansive (free, from the [spec] log lines)
core MLP sigma_max, control vs SCSE:
  step  100:  1.44 vs 1.45
  step  400:  1.87 vs 2.83
  step 1200:  2.59 vs 4.73
  step 2000:  3.03 vs 6.04   <- SCSE peak
  step 3200:  4.24 vs 4.77
SCSE's core weights blow up early (1.5x the control by step 400), peak near 6, then relax.
The spectral norms separate at steps 200-400; the loss curves separate at 250-500. The timing
matches.

## Working explanation (NOT yet tested)
Delta_0 = 0.1*(init_proj(e) - anchor_proj(e)) enters the loop at ~0.1x the anchor norm, but
MORPH's core blocks are RMSNorm-PRE-NORMALIZED: their output scale is set by learned weights
and is independent of input scale. So the first core application rescales Delta by ~80x, the
optimizer chases that, sigma_max doubles relative to control, the loop becomes expansive, and
training stalls.

If this holds, the defect is a SCALE-MATCHING problem between the SCSE entry condition and a
pre-normalized block stack -- not a refutation of source-centering as such. That distinction
matters and must NOT be blurred in the writeup.

## What would test it (candidates, none run)
* Enter at a scale the core expects: Delta_0 with norm ~ ||h*|| instead of 0.1x.
* s = 1.0 (the auditor's recommended second point; exactly MORPH's core map in deviation
  coordinates) -- checks whether the damping is implicated.
* Normalize the reconstruction, or scale-match Delta before the first block.
* Same probe on the control for a like-for-like ||carrier|| trajectory.

## Trap to avoid in the writeup
The pre-registered P1 validity gate PASSES for this arm (b_t = 0 exactly, delta0_rel > 0).
So by the gate's own definition this IS SCSE, and the negative result is a real result about
this port. Do not use "there might be a scale bug" to dodge the verdict -- report the verdict
AND the suspicion, separately.

## Evidence 3 — the replay is BIT-EXACT, and it exposed a conditioning question (2026-08-25)

Replaying one captured core step with the shipped formula, on active slots only:
  bf16 autocast : 4.85e-1  6.41e-1  6.23e-1  6.14e-1  6.04e-1
  no autocast   : 0.00e+00 0.00e+00 0.00e+00 0.00e+00 0.00e+00   <- matches the capture exactly

Two conclusions.

1. EMPIRICAL PROOF that the campaign runs the corrected recurrence. `Delta + s*(stack - Delta)`
   reproduces the training trajectory EXACTLY (0.000e+00) at fp32. This is stronger than the
   file-mtime argument. An earlier version of this test reported 0.49-0.72 error for BOTH
   formulas; that was a precision mismatch in the TEST, not a defect in the code, and it must
   not be cited as evidence of anything.

2. OPEN QUESTION, NOT YET A FINDING: recomputing the SAME step in bf16 instead of fp32 moves
   the result by ~50-64%. Training runs under `torch.autocast("cuda", dtype=torch.bfloat16)`
   (train.py:119 and elsewhere), so this is the precision the model actually computes in.
   A ~60% per-step sensitivity to rounding would mean the loop is numerically chaotic and the
   gradients are noise-dominated -- which would explain a stall.

   *** DO NOT REPORT THIS AS A CAUSE YET. *** The CONTROL has not been measured. MORPH is a
   deep looped bf16 model and the control may well show a similar number, in which case this
   says nothing about SCSE. The comparison is the whole content of the claim.

   Deferred deliberately: measuring it requires building a second model while a training run
   holds 21.3 GiB, and my first attempt OOM'd (my process only; the trainer survived, verified
   at step 200 with no errors). Not worth risking the final run for a measurement that can be
   taken 30 minutes later. Run it AFTER the campaign, one model per process.

## Standing lesson
Build ONE model per process while a campaign is live. Three sequential builds in one process
exhausted the ~4.7 GiB left over and OOM'd.
