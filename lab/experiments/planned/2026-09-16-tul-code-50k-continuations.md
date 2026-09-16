# Experiment: TUL-Code arms to 50k from their 20k checkpoints (plain, jepa, xm; ruler for pairing)

Status: planned
Date: 2026-09-16
Owner: Claude (session f9558148), for Wolfe ("we need to do xm, jepa, and plain to 50k from their 20k checkpoints")

## Question

Every TUL-Code arm stopped at 20k with phase 3 (rollout: the coda reads sampled codes) only 10k steps old, on a flat 1e-4 learning rate validated for the token model and never for the flow loss. The thinker's flow-loss share of the code's variance has a measured phase-3 slope of about −0.002 per 1000 steps on the joint arms, the paired gap against the strict ruler closed from +0.71 (5k) to +0.63 (20k), and the XM arm's best-of-4 selection ratio fell from 0.95 to 0.88. Wolfe's reading: a slow learner on a non-optimal learning rate. Do 30k more steps with a decay move the thinker's floor, the sample, and the coda's use of it?

## Hypothesis

H-50-1 (slow learner): with more steps and a decaying rate the flow share keeps falling, the sample residual leaves the 1.7–1.9× band, and the paired gap against the ruler keeps closing.
H-50-0 (null): the decay lowers the token CE on every arm including the ruler, the thinker's share moves by no more than its flat-LR slope would give, and the paired gap does not close: the copy-of-the-span target bounds the guess.

## Predictions (frozen)

Reference at 20k (one seed each): flow probe share (fresh pairs, 96 rows) parent 0.31, jepa 0.274; logged train/code_fm_rel parent 0.308, jepa 0.282, xm 0.312 (xm's is a best-of-4 seed statistic); sample residual over code variance, rank-128 head, parent 1.85 / 1.83, jepa 1.84 / 1.84; 8-draw marginal k16 − k1 parent +0.014, jepa −0.017; paired one-draw gap vs the strict ruler at 20k parent +0.626, jepa +0.654; XM selection ratio (best/mean of 4) 0.88 at 15k.

- P-50-1. Flow probe share on the 50k checkpoint ≤ (its 20k value − 0.05) on every code arm (parent ≤ 0.26, jepa ≤ 0.224, xm ≤ its 20k probe − 0.05). 40 %.
- P-50-2. Sample residual in the rank-128 head at 50k ≤ 1.65 on at least one code arm (from 1.84–1.85). 40 %.
- P-50-3. Paired one-draw gap against the 50k ruler ≤ +0.50 on at least one code arm (from +0.63). 40 %.
- P-50-4. 8-draw marginal k16 − k1 at 50k ≤ −0.05 on at least one code arm (from ≈ 0). 30 %.
- P-50-5. XM selection ratio (train/code_xm_score_best / train/code_xm_score_mean, 1000-step mean) at 50k ≤ 0.85 (from 0.88). 45 %.
- P-50-6. Ordering on the paired gap at 50k: xm ≤ jepa ≤ plain, each pair separated by a CI that excludes 0. 30 %.
- P-50-7. The plain arm's paired gap vs the ruler closes by ≥ 0.05 from 20k to 50k (the coda-side slow-learner signal). 55 %.
- P-50-8. Healthy continuations; the ruler's val CE at 50k is ≥ 0.10 below its 20k value (the decay does what decays do). 75 %.

## Binding

- P-50-1 and P-50-3 hold → slow learner confirmed; the recipe for code arms becomes the 100k horizon with a decay, and the next panel is run at that length.
- P-50-8 holds and P-50-1 fails → the decay helps the token model and not the thinker: the floor is the target's; the code must change (a smaller code, a discrete plan code).
- P-50-7 holds and P-50-2 fails → the coda keeps learning to use a sample that is not getting nearer: reader-side gains only; the thinker is at its target's ceiling.
- P-50-5 holds and P-50-6 fails → XM's selection sharpens without reaching the coda; a K sweep is not worth its cost.

## Method

Four continuations through the runner (`/home/wolfe/morph-scratch/arc/recon_arms.txt`), optimizer state carried (no fresh optimizer), each from its 20k checkpoint, common overrides `training.steps=50000 training.min_lr=1e-5 training.ademamix_t_beta3=20000 training.ckpt_every=5000` (the schedule is a cosine over `steps` from the 1000-step warmup: at the resume point, step 20000 of 50000, the rate steps from 1e-4 to about 6.7e-5 and decays to 1e-5 at 50000; a discontinuity by design):

- `tul-code-20k-50k`: config `tul_code`, resume tul-code-20k step_20000.pt (the plain code arm).
- `tul-code-jepa-50k`: config `tul_code_jepa`, resume tul-code-jepa step_20000.pt.
- `tul-code-xm-50k`: config `tul_code_xm` (K = 4, l2 selection), resume tul-code-xm step_20000.pt (queued third so that checkpoint exists when the runner reaches it).
- `strict-ruler-50k`: config `tul_slot_spandec_strict`, resume slot-spandec-strict-20k step_20000.pt; the pairing partner for P-50-3/6/7, no code.

Runner sweeps at 30000, 40000, 50000; a Spark watcher (`spark_code_probes_50k.sh`) runs the 8-draw marginal sweep at each checkpoint and the subspace probe, span samples and flow probe at 50000 on the three code arms; paired scoring against the 50k ruler with `lab/divergence/paired_vs_ruler.py`. Each arm gets its own wandb run (the fork logic gives a resumed run with a new `wandb.name` a new id). Supersedes the never-launched 30k decay pair (`planned/2026-09-16-tul-code-lr-decay.md`, removed from the queue and deleted in this change; its question is answered by the plain arm and the ruler here).

Method amendment 2026-09-16 01:30 (before any continuation started; reason: a defect
found on review). The phase boundaries are FRACTIONS of `training.steps` (`train.py`:
`code_phase2_at * total_steps`, `code_phase3_at * total_steps`), so a resume with
`training.steps=50000` and the config's 0.10 / 0.50 would put the continuation back
into phase 2 (coda on truth codes, no rollout) from 20k to 25k and re-enter phase 3 at
25k. The three code continuations therefore carry `+tul.code_phase2_at=0.04
+tul.code_phase3_at=0.2` (2000 / 10000 of 50000: the parents' own boundaries), so the
resumed run stays in phase 3 from its first step. The ruler has no phases. Also on
2026-09-16: the four queue lines the runner consumed at 01:13 were malformed (the
commit field had lost its separator) and were skipped; they were re-queued at 01:15
behind `tul-code-lejepa`, which had started in the gap. Predictions unchanged.

## Not verified before launch

- The resumed AdEMAMix state under the LR step at the resume point (never run on this recipe).
- The XM arm's 20k checkpoint does not exist at queue time; the runner's smoke and the checkpoint's existence gate its start.
- One seed per arm; the ordering prediction P-50-6 is the weakest.

## Results

(after the run)

## Verdict

(after the run)

## Updated hypothesis

(after the run)
