# Planned: the honest l2cap re-baseline — carry none, full stack, 4500 steps

Status: planned
Date: 2026-08-31 (frozen before the 450-step smoke and the full run; Wolfe's
approval verbatim: "you have may approval to run this for full training after
the 450 step smoke. make sure youre not eating the disk up with checkpoints").

## Question

With the retention-carry leak fixed (`retention_carry: "none"` default, commit
01cf079) — does the l2cap recipe (full BPTT + σ≤1.5 projection + TUL slot
geometry + mux) earn ANY honest depth? Every prior depth number is void: the
carry-off audit showed the 4500-step "0.233 earned" inverts to −1.12 without
the leak. This run is the first model TRAINED causal under the recipe, on the
merged stack (causality fix + gathered slot attention + compile_blocks).

## Arm

`tul_l2` + `training.compile_blocks=true`, panel flags (steps=4500 batch=6
seed=1 alpha_cap=3.5 t_beta3=3500 use_kernels=false eval_ablations=true
eval_every=250 gen_every=0 ckpt_every=500 grad_probe_every=1), wandb name
`tul-l2nc`. Checkpoints pruned to step_4500 after the run (disk directive).
The depth sweep on this model IS the honest sweep — there is no carry to cut
(the audit's standing carry-off control is satisfied by construction).

## Predictions (frozen)

- **P1 (binding — loop formation).** Depth-earned CE (K1 − K6, 48-row auto
  sweep at step 4500) ≥ 0.10 nats: **25%**. Prior lean AGAINST: the only arm
  that ever showed earning was leak-driven, and every capped/truncated arm was
  flat; but no arm has ever TRAINED causal, so formation was never cleanly
  tested — full BPTT's gradient now has only honest channels to use.
- **P2 (honest CE level).** val/ce_tokens@4250 in [4.30, 4.90]: 70%. The old
  4500 K1 (carry-free) was 4.622; training without the carry should land the
  model near or below its old honest number, not above it.
- **P3 (stability).** S1-clean (no eval > running-min + 0.20 ×2 consecutive
  after step 1000): 85%.
- **P4 (speed).** Full-step sps at the 450-smoke ≥ 2.0 (vs 1.31 eager asis,
  1.55 cattn): 75%. fwd+bwd measured 1.84×; optimizer+projection+eval overhead
  dilutes it by an unmeasured amount.
- **Decision rules (binding).** P1 TRUE ⇒ loop formation is real and was
  masked by the leak — the loop program continues on the causal base (G3/G4
  next, honest ledger opens). P1 FALSE ⇒ the recipe earns nothing honestly;
  the loop-formation program pauses and TUL's value case reverts to
  conditional-compute only — the honest 30k head-to-head becomes the next
  experiment. P4 FALSE ⇒ compile_blocks stays opt-in and the discrepancy vs
  the bench gets its own investigation before any long run uses it.

## Method

1. 450-step smoke, gates: "BLOCKS compiled" print; "Core spectral PROJECTION
   ON"; "TUL ON"; NO acausal-carry warning; loss@400 < 14; zero mid-run
   recompiles; step_450 checkpoint LOADS via tul_samples.load_ckpt (the
   block-level `_orig_mod` prefix round-trip). Any gate fails ⇒ abort, no full
   run, investigate.
2. Full 4500 run, then: 48-row depth sweep (auto), tul_samples at step_4500,
   prune checkpoints to step_4500 only.
3. Artifacts → lab/experiments/results/2026-08-31-l2-honest-rebaseline/;
   wandb adew-me/morph-tul run tul-l2nc.

## Not verified before launch

compile_blocks has never run a full training step with optimizer + projection
(fwd+bwd only); the causal default has never trained (that is the experiment);
the merged stack (fix + gathered attention + compile_blocks) is exercised
together for the first time in the smoke.
