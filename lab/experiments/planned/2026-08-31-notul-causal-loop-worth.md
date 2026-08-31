# Planned: honest token-path loop contribution — causal no-TUL MORPH, 4500 steps

Status: planned
Date: 2026-08-31 (frozen before launch; Wolfe: "let's check a MORPH with out
tul for loop contribution").

## Question

Does the TOKEN-path core loop (every position loops, no slot geometry) earn
its iterations when trained causal? No honest reading exists: the historical
"core loop worth 0.017 nats" was measured under the base.yaml acausal carry,
and the bench-450 token sweep (K1 15.53 → K6 7.43) is both acausal-trained
and K1-OOD (per-sequence Poisson(6) training rarely samples depth 1). This is
the token-path twin of the l2nc re-baseline, and tests whether TUL's slot
geometry was what suppressed loop utility.

## Arm

`notul_l2` (l2cap recipe, TUL off, kernels on; inherits the post-fix
`retention_carry: "none"`), panel flags (steps=4500 batch=6 seed=1
alpha_cap=3.5 t_beta3=3500 eval_every=250 ckpt_every=500 gen_every=0
grad_probe_every=1), wandb `notul-l2nc`. ~59 min at the benched 1.28 sps.
Post-run: `token_depth_sweep.py` depths 1..8, 48 rows; plain gen samples;
checkpoints pruned to step_4500.

## Predictions (frozen)

- **P1 (binding).** Trained-support earning CE@K3 − CE@K6 ≥ 0.10 nats: 20%.
  Prior: flat, like every honest loop reading — but the token path has never
  actually been read honestly, and all 1152 positions getting the full core
  is a different regime from 64 slots.
- **P2.** OOD-shallow spread: CE@K1 − CE@K3 ≥ 0.5 nats: 70% — depth 1 is
  undertrained for a Poisson(6) model, so a large K1 penalty is expected and
  is NOT loop contribution; recording it as its own number keeps the two
  effects separate.
- **P3.** S1-clean over 4500: 85%.
- **Binding.** P1 TRUE ⇒ the token-path loop composes where slots do not —
  major; reopens formation via geometry, own Agent Note, and the honest 30k
  head-to-head design gets revisited (B would out-earn A). P1 FALSE ⇒ the
  loop is honestly flat in BOTH geometries; "no reliable way to make the core
  contribute at this scale/task" is the standing conclusion and the
  conditional-compute head-to-head proceeds unchanged.

## Not verified before launch

notul_l2 has never trained with carry "none" (smoke gates cover composition:
kernels=fused, projection armed, no TUL, no acausal warning); the causality
contract on the kernel path is covered by tests only for the eager path —
the fused-GLA path's carry wiring goes through the same track_ret sites, but
no fused-path corruption probe has run (flagged; the eager/fused parity gate
in gla.py is the standing evidence).
