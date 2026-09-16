# Experiment: does a learning-rate decay lower the thinker's floor? (parent + ruler, 20k → 30k)

Status: planned
Date: 2026-09-16
Owner: Claude (session f9558148), for Wolfe ("this looks like it is trending down to me.
But this is not an optimal learning rate either way.")

## Question

The thinker's flow loss on every joint arm sits at 0.28–0.33 of the code's variance from
step 4000 on, with a measured phase-3 slope of −0.0017 per 1000 steps on the parent
(−0.0020 on jepa, −0.0006 on thinker-only). Every arm ran the winner recipe's FLAT
1e-4 (cosine with `min_lr == lr`), validated for the token model, never for the flow
loss. A conditional-flow-matching loss has a large gradient-noise floor (random t, random
noise pairing per step), and a flat LR holds the parameters at that noise level. Does a
decay lower the thinker's floor, and does the coda's paired gap against the ruler move
with it? Two continuations from the 20k checkpoints, same decay on both so the pairing
stays matched.

## Hypothesis

H-L1: the plateau is an LR-noise floor; a decay to 1e-5 drops the thinker loss by
several hundredths within 10k steps and the sample gets nearer the truth.
H-L0: the plateau is the target's unpredictable share; the decay lowers the token CE
(as any decay does) but the thinker's share moves by no more than its flat-LR slope
would have (≤ 0.02), and the paired gap against the ruler does not close.

## Predictions (frozen)

Reference: parent flow share 0.308 at 20k (`train/code_fm_rel`, 1000-step mean),
band0 0.46, band3 0.22; paired one-draw gap vs the ruler +0.626 at 20k; ruler val CE
falls ~0.05 per 10k at flat LR (its 15k→20k drop).

- P-L1. Parent `train/code_fm_rel` (1000-step mean) at 30k ≤ 0.27 (a 0.04 drop, twice the
  flat-LR extrapolation). 45 %.
- P-L2. Flow probe share on the 30k checkpoint (fresh pairs) ≤ 0.28 (from 0.31). 45 %.
- P-L3. Paired one-draw gap vs the DECAYED ruler at 30k ≤ +0.55 (from +0.626 at 20k,
  both arms with the same schedule). 40 %.
- P-L4. Sample residual in the rank-128 head at 30k ≤ 1.70 (from 1.85). 40 %.
- P-L5. Both continuations healthy; the ruler's own val CE at 30k is ≥ 0.08 below its 20k
  value (the decay does what decays do on the token model). 75 %.

## Binding

- P-L1 and P-L3 hold → the LR was a real lever; every future code arm decays, and the
  slow-learner reading stands: re-plan the runs at the 100k recipe with a decay.
- P-L5 holds, P-L1 fails → the decay helps the token model and not the thinker: the
  floor is the target's, not the optimizer's; the code must change (XM, bottleneck).
- P-L1 holds, P-L3 fails → the thinker fits the copy better and the coda cannot use it.

## Method

Two resumes with the runner (`recon_arms.txt`), optimizer state carried (no fresh
optimizer):

- `tul-code-20k-decay30k`: config `tul_code`, `training.resume` = tul-code-20k
  `step_20000.pt`, `training.steps=30000`, `training.min_lr=1e-5`,
  `training.ademamix_t_beta3=20000`, `training.ckpt_every=5000`.
- `strict-ruler-decay30k`: config `tul_slot_spandec_strict`, `training.resume` =
  slot-spandec-strict-20k `step_20000.pt`, the same overrides.

The schedule is a cosine over `steps` from the 1000-step warmup, so at the resume point
(20000 of 30000) the LR steps from 1e-4 down to about 3.4e-5 and decays to 1e-5 at 30000:
a discontinuous drop by design (the cheapest decay test; not a tuned schedule). Runner
sweeps at 25000 and 30000; paired scoring at 30000 against the decayed ruler
(`paired_vs_ruler.py`); flow probe and subspace probe on the parent's 30000 checkpoint on
the Spark; 12-step runner smoke as the gate. Queue position: after the running
`tul-code-xm`, before `tul-code-xmc`.

## Not verified before launch

- The resumed optimizer under the LR step: the AdEMAMix state carries over (the thinker-
  only fix for uint8 codes applies); the abrupt LR drop has not been run on this recipe.
- One seed per arm.

## Results

(after the run)

## Verdict

(after the run)

## Updated hypothesis

(after the run)
