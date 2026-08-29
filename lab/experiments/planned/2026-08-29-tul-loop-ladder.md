# Planned: the loop ladder — L1 full-BPTT / L2 projected / L3 DB-shaped / L4 AdamW

Status: planned
Date: 2026-08-29 (frozen before smoke; smoke and launch deferred until the GPU
is free). Configs: `tul_l1.yaml`, `tul_l2.yaml`, `tul_l3.yaml`, and L4 =
`tul_l1` + `training.optimizer=adamw`. Decision note:
`.agents/notes/proposed/architecture/2026-08-29-loop-ladder.md`.

## Questions

- **Q-loop.** Does restoring the core loop over the slot write (full BPTT,
  nothing protecting it) destroy the GL mechanism — and if it does, do the
  probes show instability signatures (σ_max runaway, compounding core_gain) or
  quiet homogenization (slot_pairwise_cos climbing at healthy norms)?
- **Q-control.** If L1 shows instability, does the hard σ ≤ 1.5 projection
  (L2) restore a working write at the same architecture?
- **Q-local.** Does the DB-shaped loop (L3: no gradient across iterations,
  per-iteration local mux) keep the write load-bearing with a looped forward?
- **Q-optim (Wolfe's).** Is the instability AdEMAMix-at-batch-6, not the loop?
  L4 differs from L1 only in `training.optimizer=adamw`.

## Reference numbers (fixed)

gl1b-s1 4.4047 (worth 0.05–0.096) | gl1b-s2 4.3714 (worth ≤ 0.028) |
gl1b-nomask 4.3102 | gl1-ctrl 4.6656. All `val/ce_tokens` @4250, batch 6,
4500 steps. Note the reliance metric's seed fragility (R1 of the line-2
filing): single-arm worth numbers are read as bands, not points.

## Predictions (frozen)

- **P1 (L1).** Completes without detonation (S1 holds, no NaN): 60%. Sustains
  worth_shuffle ≥ 0.04 (≥3 consecutive evals incl. one of the last three): 25%
  — my prior says the write homogenizes (slot_pairwise_cos > 0.4 by step 4500)
  even when training is numerically stable. CE @4250 ≤ 4.45: 40%.
- **P2 (L2).** Conditional on L1 showing instability signatures (σ_max > 3.0
  at any log, or per-iteration core_gain compounding with S1 violation): L2
  removes the signature AND sustains worth ≥ 0.04: 60%. If L1 is clean, L2
  lands within noise of L1 (projection inert at healthy σ): 80%.
- **P3 (L3).** Completes clean: 90%. worth ≥ 0.03 sustained: 55%. CE @4250 ≤
  4.46 (within 0.05 of gl1b-s1): 50%.
- **P4 (L4, Wolfe's hypothesis).** Decision rule, binding: if L1 violates S1
  (or aborts) with instability signatures AND L4 passes S1 with worth ≥ 0.04
  ⇒ optimizer-causal — AdEMAMix-at-batch-6 is implicated and the fatal-loop
  framing is dead. If L1 and L4 fail the same way ⇒ loop-side. If both pass
  S1 ⇒ no instability at this scale; the ladder is scored on worth/CE alone.
  My probability that L4 is strictly more stable than L1 (lower max σ_max,
  lower max core_gain, no S1 violation): 55%. Wolfe's stated position: most of
  the instability is the optimizer at this batch size.
- **S1 (stability, all arms).** No run sits > 0.20 nats above its running min
  for 2+ consecutive evals after step 1000.
- **F (the falsifier, binding).** ANY full-BPTT arm (L1, L2, L4) sustaining
  worth ≥ 0.04 at CE ≤ 4.50 falsifies "gradient through an iterated write is
  fatal", and the campaign docs that state it get corrected in the filing
  change.

## Method

Sequential, one trainer on the 5090, ~30-35 min each (the loop adds ~22M core
params and per-iteration compute on the compact slot sequence):

1. smoke: 30 steps of L1 and L3 (the two new code paths; L2/L4 are config-only
   deltas on L1), gating VRAM and NaNs. Smoke AFTER this file is committed.
2. `tul_l1` / `tul_l2` / `tul_l3` / `tul_l1 + training.optimizer=adamw
   wandb.name=tul-l1-adamw`, each with the panel flags: steps 4500, batch 6,
   seed 1, ademamix_alpha_cap 3.5, ademamix_t_beta3 3500 (inert under adamw),
   use_kernels false, eval_every 250, gen_every 0, ckpt_every 500,
   grad_probe_every 1 (per-iteration loop probe), tul.eval_ablations true.
   σ_max logs every 100 steps by default (spectral_penalty_log_every).
3. Prune each run's checkpoints to step_4500 after its verdict; artifacts to
   `lab/experiments/results/2026-08-29-tul-loop-ladder/`.

**Method amendment (2026-08-29 13:50, before any arm completed).** First launch
aborted: tul-l1 OOM'd at step-0 eval inside `tul_mux_grad_share` — the one
grad-enabled eval instrument, which runs with eval-mode checkpointing OFF and
so retained the full-BPTT loop's activations at batch 6. Fix (commit 28549d0):
the probe slices to 2 rows. The smoke gate is amended to steps=12/eval_every=5
so the eval path is exercised (the no-eval smoke waved the bug through).
Predictions are UNTOUCHED. tul-l1's 3-minute aborted run (yzmy9jli) is
discarded as a harness failure, not evidence about any hypothesis.

## Not verified before launch

The n_core>0 + tg_restrict + slot_seed=boundary + mux composition has never
run on GPU (CPU tests only: 683 passed, 2 xfailed at commit time). The db_loop
memory profile (db_mux_iters=4 × [B,S,V] fp32 logit graphs ≈ +0.3 GB) is
arithmetic, not measurement — the smoke gates it.
