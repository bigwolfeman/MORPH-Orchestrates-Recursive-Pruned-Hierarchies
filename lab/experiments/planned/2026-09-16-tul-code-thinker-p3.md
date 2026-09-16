# Experiment: LCTUL, long fixed-target phase 2, then unfreeze the coda on samples only

Status: planned
Date: 2026-09-16
Owner: Claude (session f9558148), for Wolfe ("I think the way it should work is: long
phase 2, phase 3 is just unfreezing the coda, no ground truth latents.")

## Question

Every joint LCTUL run trained the coda on a 50/50 mix of noised truth codes and sampled
codes in phase 3, with the encoder, prelude and coda all still moving under the CE, and
the thinker chasing a target that moved with them. The thinker-only arm gave the
thinker a FIXED target for 40k steps (E + prelude + coda frozen at the parent's step
10000) and never rolled out; on that frozen, trusting coda the sample decodes to word
salad (val loss 12.8 nats with every slot sampled, ce_marginal 9.37). This arm joins
the two halves the way Wolfe describes: resume that checkpoint, unfreeze the coda and
its head path, keep the thinker training on the still-fixed target, and hand the coda a
sampled code at EVERY valid slot. No truth latent reaches the coda from the first
resumed step. Does a coda that must read samples learn to read what the thinker puts in
them, or does it learn to ignore the cells (rollout1's ending: ce_tf 4.03, the ruler)?

## Hypothesis

H-P3-1 (Wolfe): the thinker's sample carries the context in a form the frozen phase-2
coda cannot parse; a coda trained on samples alone learns the sample's dialect and the
gap to the ruler closes.
H-P3-0: the sample carries nothing the coda can use; the cheapest coda is one that
ignores the cells, and the arm converges to the ruler with ce_tf rising to the no-code
level.

## Predictions (frozen)

Reference: the thinker-only arm at 50k, flow probe 0.279, residual 1.72, sampled val
12.8 / ce_tf 0.41; rollout1 at 20k: ce_tf 4.03, paired gap vs the ruler ≈ 0; the parent
at 20k: paired gap +0.63, ce_tf 0.35.

- P-P3-1. `val/ce_tokens` (every slot sampled) falls below 4.6 within 5000 resumed steps
  (the coda recovers the token-path level either way). 90 %.
- P-P3-2. `val/ce_tf` (truth codes on the unfrozen coda) at the end (step 70000) ≥ 3.0:
  the coda has gone cell-blind. 65 %.
- P-P3-3. Semantic probe (`code_semantic_probe.py`, 120 cuts) at 70000: OWN − SHUF
  cosine to the true span > 0 with the CI excluding 0. 25 %.
- P-P3-4. 8-draw marginal k16 − k1 at 70000 ≤ −0.05. 25 %.
- P-P3-5. Paired one-draw CE against the parent at 20k on the same 480 rows: the arm is
  ≥ 0.10 nats BETTER (the sample-only coda beats the mixed coda). 30 %.
- P-P3-6. Healthy; rate ≥ 20k tok/s at batch 6; no rank abort (E is frozen; rank cannot
  move). 80 %.

## Binding

- P-P3-2 fails and P-P3-3 or P-P3-5 holds → the mixed phase 3 was the defect: the coda
  can learn the sample's dialect when nothing else is offered. Re-run the joint recipe
  with a frozen-target phase 2 and a samples-only phase 3 from scratch.
- P-P3-2 holds → H-P3-0: with nothing in the sample to read, the coda drops the cells.
  The remaining lever is the code's definition, not the phase schedule.

## Method

`morph/configs/tul_code_thinker_p3.yaml`: resume `tul-code-thinker/step_50000.pt` with a
fresh optimizer over `train_only` = the thinker (core, core injections, injection,
tul_code_head/time/cell/clean, as `tul_code_thinker.yaml`) + coda, coda injections
(x0_injects.10–13), lm_mixer, final_norm; E, prelude, embeddings, the slot module frozen.
`code_phase2_at 0.0`, `code_phase3_at 0.0`, `code_rollout_p 1.0`, batch 6, 20000 steps
(50000 → 70000), ckpt every 5000, `ademamix_t_beta3 20000`. Runner sweeps at 55k…70k;
Spark: marginal sweep at each checkpoint, subspace + flow + span samples + semantic probe
at 70000. Pairing: the parent's 20k sweep on the same rows.

## Not verified before launch

- The 12-step Spark smoke of the resume (train_only 117.4M trainable / 152.6M frozen,
  phase 3 from the first resumed step) — its exit code is the gate before the queue entry.
- The frozen embedding table is the tied LM head, so the coda adapts through its blocks,
  injections, mixer and final norm only.
- One seed.

## Results

(after the run)

## Verdict

(after the run)

## Updated hypothesis

(after the run)
