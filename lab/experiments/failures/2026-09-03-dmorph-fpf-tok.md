# Experiment: dmorph v1.1 — Fixed-Point Forcing on the tok arm

Status: failure

Design: `.agents/notes/rejected/architecture/2026-09-03-fixed-point-forcing-for-dmorph-and-the-loop.md`.
Paper: Flow Reasoning Models (Helbling et al., arXiv 2606.29150), mirrored at
`docs/references/training-objectives/flow-reasoning-models/`.
Code: `morph/model/dmorph.py` (`carry_in`, `integrate`, `fpf_rollout`, `residual_auroc`),
config `morph/configs/dmorph_tok_fpf.yaml`, tests `tests/test_dmorph_fpf.py`.

## Question

Does training the noisy stream on carries from its OWN rollout (FPF, no gradient through
the rollout) turn the dmorph ladder from a depth-inverted iteration into one that beats
its own one-pass head and keeps improving with held-time depth — on text, where the
paper never ran?

## Hypothesis

The v1 ladder's failure (dmorph-tok-s1-5k: `ladder_ce` 7.25 vs one-pass `dm_ce` 4.94
at step 1500) is the paper's exposure bias: the stream never sees its own predictions
in training. A carry trained on rollout states removes most of the gap; held-time
recurrence at inference then adds a little; the convergence residual becomes
informative. The clean head (the metric) is not harmed.

## Predictions (frozen before launch)

Reads: the mean of the last three evals of a run (steps 4500, 4750, 5000; each eval is
`n_eval_batches` batches at seq 1024 batch 6), the fpf arm `dmorph-tok-fpf-s1-5k`
against the v1 arm `dmorph-tok-s1-5k` (same seed, same steps, same rows, source_std 1.0).

- P1 (65 %): fpf `dm_ladder_ce_r0` is at least 1.0 nat BELOW v1 `dm_ladder_ce`.
- P2 (55 %): fpf `dm_ladder_ce_r0` is within 0.3 nats of fpf's own one-pass `dm_ce`
  (the null-carry one-pass read, same eval).
- P3 (50 %): held-time depth helps: fpf `dm_ladder_ce_r2` ≤ `dm_ladder_ce_r0` − 0.02.
- P4 (55 %): fpf `dm_resid_auroc_r2` ≥ 0.65 (paper Fig. 6c: 0.50 without FPF, 1.00 with).
- P5 (70 %): the fpf clean head `val/ce_tokens` is within ±0.05 of v1's.
- P6 (70 %): fpf throughput (`tok/s`, mean over steps 1000–5000) ≥ 0.85 × v1's. Analytic:
  the rollout is ≤ one flat-stack forward on half the rows, no backward, ≈ +5 % on the
  training step.

Success = every prediction resolves on its majority side. One seed: the run compares two
runs at the same seed and MORPH runs decorrelate in 11 steps (memory), so the P1 margin
of 1.0 nat is set far above the 0.085-nat per-eval spread; P3's 0.02 is within noise and
is the weakest read here by design (it is a within-run, same-rows comparison, which is
the low-noise kind).

## Method

`bash lab/dmorph/run_panel.sh` with `DMORPH_ARMS=dmorph_tok_fpf DMORPH_SEEDS=1
DMORPH_STEPS=5000 DMORPH_TAG=-5k` after the v1 panel's hs run releases the card (ONE
trainer). Everything else is the v1 tok arm's recipe (`dmorph_tok.yaml` +
`dmorph.source_std 1.0`, Method amendment 2 of the v1 panel), plus `dmorph.fpf_p 0.5`,
`dmorph.recur 0` (the paper's training integrator is the plain self-conditioned Euler
ladder; held-time updates are an eval dial here, r ∈ {0, 2}). Abort rule: the trainer's
divergence guard (`preclip/total > 1e4` after step 200, on `loss_tokens_only`).

Artifacts: `lab/experiments/results/2026-09-03-dmorph-v1-panel/dmorph-tok-fpf-s1-5k.log`
(same directory as the v1 panel it is read against) and the wandb run of that name in
`adew-me/morph-tul`.

## Results (2026-09-03 15:50)

`lab/experiments/results/2026-09-03-dmorph-v1-panel/README.md`, "The FPF run". Last-3-eval
means, FPF vs v1 tok: clean head 4.3920 vs 4.3588; one-pass 3.8251 vs 3.8027; ladder r0
6.7974 vs 6.3956; ladder r2 6.5226; residual AUROC r2 0.4678; tok/s 17,104 vs 19,997
(0.855×). The rollout ran on 47.4 % of rows (mean `t_start` 0.224) and the carry projection
trained to norm 24.1 by step 2500.

## Verdict

P1 FALSE (the FPF ladder is 0.40 nats WORSE than v1's, on every eval from step 1250), P2
FALSE, P3 TRUE (two held-time updates buy 0.27 nats), P4 FALSE (AUROC below chance on 15
of 16 evals), P5 TRUE, P6 TRUE → failure.

## Updated hypothesis

On text the rollout carry is learned and used but makes the ladder worse, not better, and
a carry that has stopped moving marks a WRONG position slightly more often than a right
one — the opposite of the paper's Fig. 6c. The paper's premise is a single correct answer
per position that the recurrence can converge to; a text position has a distribution, and
"the carry has converged" there means the stream has committed to one token, which is
exactly the overconfident fixed point FPF was supposed to remove. Held-time depth still
helps a little (P3), so the recurrence is not inert; it is aimed at the wrong target. The
next honest test is not more FPF on this ladder but a target with one answer — the FM
planner's span-level plan or the hs slot state — with the residual AUROC as the gate, and
only if someone still wants a decodable-carry recurrence in MORPH at all.

## What the method could not distinguish

One seed, 5k steps, `recur 0` in the rollout (the paper's training integrator) — a rollout
with held-time updates was not run. The carry form (`normalize(D̂)`, not the categorical
prediction) is a deviation the run cannot separate from the text-vs-puzzle difference.
Throughput passed P6 by 0.005; the wall clock (35.5 vs 29.4 min, compile included) says
0.83×.
