# Experiment: LeJEPA on the code — full flow gradient into the encoder with SIGReg as the guard

Status: failure
Date: 2026-09-16
Owner: Claude (session f9558148), for Wolfe ("that sounds like we should be using lejepa sigreg?")

## Question

The jepa arm let the flow loss's gradient reach the encoder at one tenth strength with a
rank abort as the only collapse guard: the code's rank slid 70 → 19 → 31, the flow share
fell 0.31 → 0.27, the phase-2 one-step sample read 2.5 nats better on the trusting coda,
and the code stayed a verbatim copy (ce_tf 0.32). LeJEPA (Balestriero & LeCun, arXiv
2511.08544) is the principled form of that arm: the prediction loss trains the target
encoder in full with no stop-gradient, and SIGReg (random 1-D projections tested against
N(0, 1) with the Epps–Pulley statistic) holds the embedding distribution at an isotropic
Gaussian, which forbids dimensional collapse by construction. Does the full gradient plus
the isotropy guard give a code that is predictable from the past AND still decodable,
where a tenth of the gradient plus a rank tripwire gave a lower-rank copy?

## Hypothesis

H-LJ1: with the full gradient the encoder moves the code toward what the past determines;
SIGReg keeps its rank full so the coda keeps its content; the thinker's flow share falls
well below jepa's 0.27 and the sample residual leaves the 1.7–1.9× band.
H-LJ0 (null): SIGReg holds the rank but the coda's CE anchors the copy; the flow share
lands near jepa's, the residual does not move, and the paired gap stays at +0.65.

## Predictions (frozen)

Reference at 20k (one seed): jepa flow probe 0.274, rank 31 (min 16), ce_tf 0.32, residual
(rank-128 head) 1.84, paired one-draw gap vs the ruler +0.654, sampled val 4.39; parent
0.31 / 38–48 / 0.35 / 1.85 / +0.626.

- P-LJ1. `val/code_eff_rank` stays ≥ 40 at every val from step 3000 on (SIGReg holds the
  rank where jepa's fell to 16). 65 %.
- P-LJ2. Flow probe share at 20k ≤ 0.22 (from jepa's 0.274). 40 %.
- P-LJ3. ce_tf at 20k between 0.5 and 2.5 (the code stops being a verbatim copy but stays
  decodable; jepa 0.32, no-code ruler ≈ 4.4). 35 %.
- P-LJ4. Sample residual, rank-128 head, at 20k ≤ 1.60. 35 %.
- P-LJ5. Paired one-draw gap vs the strict ruler at 20k ≤ +0.50 (from +0.654). 35 %.
- P-LJ6. Healthy to 20k; rate ≥ 30k tok/s (SIGReg on ≈ 200 codes per cell per step with
  1024 directions is cheap); no rank abort. 75 %.

## Binding

- P-LJ1 and P-LJ3 hold and P-LJ5 holds → LeJEPA is the code definition; sweep
  `code_sigreg_lambda` (0.02, 0.2) and re-run the XM rule on top of it.
- P-LJ1 holds and P-LJ3 fails (copy) → the CE anchor wins over the full gradient: the
  code's capacity must be cut by construction (fewer cells/dims, higher noise floor).
- P-LJ1 fails → λ 0.05 is too weak against the CE anchor at this batch; λ 0.2 before any
  other change.

## Method

`morph/configs/tul_code_lejepa.yaml` = `tul_code` + `code_noise_renorm: true`,
`code_target_lambda: 1.0` (the flow loss's gradient reaches E in full; the tape stays
detached), `code_sigreg_lambda: 0.05` (paper's default), `sigreg_slices: 1024` (paper's
recommendation), `code_rank_abort: 4.0` (kept as a tripwire). SIGReg
(`morph/model/sigreg.py`, Algorithm 1: 17 trapezoid knots on [−5, 5], mean over
directions, fresh directions each step) is applied per cell index to the valid slots'
codes (`tul.code_sigreg_lambda`, spec §9), exposed as `code_sigreg` / `code_sigreg_weighted`
and subtracted by the trainer so train/loss stays the CE. From scratch, the panel
settings (seq 1024, batch 6, 20k, phases 2k / 10k). Runner sweeps at 5k/10k/15k/20k; the
Spark watcher runs the marginal sweep at each checkpoint and the subspace, samples and
flow probes at 20k; paired scoring vs the ruler and the parent at 20k. Queue position:
after the four 50k continuations, before `tul-code-xmc`.

## Not verified before launch

- SIGReg's scale against the CE at batch 6: about 200 valid codes per cell per step is
  below the paper's batch of 128 images × views only in the sense of one view; the
  gradient bound is per sample and the term is bounded, but λ 0.05 has never been run on
  this model (the 2026-08-27 slot-state arm at λ 0.02 died at step 2040 on a different
  loop for other reasons).
- The interaction of a full-gradient target with the phase-3 rollout (the target moves
  while the coda reads samples): the rank tripwire is the only guard.
- One seed.

## Results

`tul-code-lejepa` (wandb pl6yf9b8, dd78b0f, 20k steps, one seed, healthy, no rank abort).
Artifacts: `lab/experiments/results/2026-09-16-tul-code-lejepa/` (flow probe, subspace
probe, 8-draw marginal at 5k–20k, span samples, the 20k depth sweep, paired scoring vs the
strict ruler and the parent, the wandb series).

| prediction | bar | LeJEPA at 20k | parent / jepa | verdict |
|---|---|---|---|---|
| P-LJ1 `val/code_eff_rank` ≥ 40 from 3k | ≥ 40 | min 40.4 (at the rollout onset), 73 at the last val, 87 in phase 2; probe participation ratio 90 / 94 | 38–48 / 16 min | hold |
| P-LJ2 flow probe share (fresh pairs) | ≤ 0.22 | 0.256 (bands 0.470 / 0.270 / 0.152 / 0.130) | 0.31 / 0.274 | fail |
| P-LJ3 ce_tf in [0.5, 2.5] | | 1.49 (probe), 1.60 (sweep k = 0) | 0.35 / 0.32 | hold |
| P-LJ4 sample residual, rank-128 head | ≤ 1.60 | 1.95 / 1.95; full 1.98 / 1.98 | 1.85 / 1.84 | fail |
| P-LJ5 paired one-draw gap vs strict ruler | ≤ +0.50 | +0.681 [+0.672, +0.691] at k = 6; +0.773 at k = 1 | +0.63 / +0.654 | fail |
| P-LJ6 healthy, ≥ 30k tok/s, no abort | | healthy; 30.1k in phases 1–2, 23.8k in phase 3 (the parent 24.0k there); peak 15.4 GB | | rate fails by the letter (the bar ignored the rollout cost) |

Other readings. Paired against the parent: +0.038 at k = 6, +0.130 at k = 1. The one-draw
depth curve has a step the family has not shown before: k1 4.585, k2 4.486, then flat to
4.501 at k16 (the 8-draw marginal single-draw mean agrees, 4.481 → 4.402); under the
marginal itself k16 − k1 is −0.0006 [−0.004, +0.003]. The code's content sits in the top
128 of 1024 directions per cell (ce_tf with the rank-128 head 1.16, with the full code
1.59: the tail HURTS the coda), and SIGReg spread variance into the tail (cumulative
variance at rank 32 is 0.47 against the parent's ~0.9). The oracle line in the span
samples is no longer verbatim: "Nief, Belgian Coordinator Gristopher Cherp gate, a doctor
who was at the hospital with 60 white Belgian" for the true "However, Belgian Chief
Coordinator Geert Gijs, a doctor who was at the hospital with 60 Belgian medical
personnel". The greedy and sampled lines are unthreaded newswire as on every arm.

## Verdict

Failure: P-LJ2, P-LJ4 and P-LJ5 fail; P-LJ1 and P-LJ3 hold. No binding line fires (the
first needs P-LJ5, the second needs P-LJ3 to fail, the third needs P-LJ1 to fail). The
full flow gradient into E with SIGReg as the guard does what it was built to do: the rank
holds, the code stops being a verbatim copy, and the thinker's flow share is the lowest in
the family (0.256). It does not make the sample a conditional draw (residual 1.95–1.98,
the most independent yet) and the coda pays for the softer code (+0.04 nats against the
parent, +0.68 against the ruler).

## Updated hypothesis

Letting the target move toward the field lowers the flow loss by moving the TARGET, not
by teaching the field the past: the encoder gives up the parts of the span the field
cannot guess, the coda loses them, and what remains is still sampled unconditionally.
The two definitions of the code, what the coda needs and what the past determines, pull
apart on web text at this span length, and a weight between them (λ on the target, λ on
SIGReg) trades one for the other rather than finding a code that is both. The k1 → k2
step is the first one-draw depth signal after rollout in the family and should be read
on a second seed before it is called anything; the next arms on this line are the
noise-search XM (`tul-code-xmn`) and, if the 50k continuations move nothing, a code with
less capacity by construction.
