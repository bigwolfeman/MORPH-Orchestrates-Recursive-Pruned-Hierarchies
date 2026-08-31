# Planned: honest token-path loop contribution — causal no-TUL MORPH, 4500 steps

Status: success — P1 landed on the prior side (loop honestly flat in both geometries); P2 missed informatively
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

## Results (2026-08-31; artifacts results/2026-08-31-notul-causal-loop-worth/; run 52.9 min, 1.45-1.49 sps)

Token-path depth curve (48 rows, paired):
K1 4.4640, K2 4.3835, K3 4.3592, K4 4.3488, K5 4.3453, K6 4.3439, K7 4.3441, K8 4.3475.

- **P1 FALSE (prior side, 80% → held).** Trained-support earning K3−K6 =
  **+0.0153** nats. With slots (l2nc): 0.006. The core loop is honestly flat
  in BOTH geometries.
- **P2 FALSE — a genuine miss (predicted TRUE at 70%).** The OOD-shallow
  spread K1−K3 is only **+0.105**, not ≥0.5. The bench-450 acausal model's
  ~8-nat K1→K2 cliff was therefore mostly the CARRY (absent at K1, feeding
  K2+), not undertrained shallow depth. Every dramatic depth effect ever
  measured in this project traces to the leak.
- **P3 TRUE.** S1-clean: monotone val/loss 5.66→4.29 across every observed
  eval (wandb sampled history returned 4 of the periodic points + the final;
  all monotone, worst excursion 0.0000).

Bonus paired observation (both arms causal, same seed/steps/batch):
no-TUL val CE 4.294 vs TUL 4.390 — token-matched, no-TUL is 0.096 nats
better (every token gets the core); TUL is 1.7× faster (2.50 vs 1.45 sps).
A 4500-step preview of the honest head-to-head's two axes.

## Verdict

Binding rule executed: the loop is honestly flat in both geometries — "no
demonstrated way to make the core contribute at this scale/task" is the
standing conclusion; the conditional-compute head-to-head proceeds unchanged.

## Updated hypothesis

Depth utility is not blocked by slot geometry. The capped core map is
contractive and hits its fixed point by ~K5 in both geometries (curves flat
past K5, slightly up at K8). If loop contribution is ever to appear, the
lever is the TASK (distributions where iteration pays), not the wiring.
