# Planned: ilv50 + l2cap_cond — the revived pair (Wolfe's call, new questions)

Status: planned
Date: 2026-08-30 (frozen before launch; GPU currently in use by Wolfe — launches
only on his word). The dbfix-pair decision rule cancelled these arms; Wolfe
revived them explicitly ("ilv50 and l2cap cond are both worth doing"). Their
questions have CHANGED since the original program note, because we now know pure
one-pass DB earns no depth (ladder inverted) and iter-conditioning alone wakes
nothing. Configs unchanged from build commit 81a9674: `tul_ilv50.yaml` (l2cap
recipe + step_mix {bptt:1, db1:1} + sigma conditioning), `tul_l2cap_cond.yaml`
(l2cap + iter conditioning ONLY).

## Question

ilv50: the bptt half of the mix is the full l2cap recipe (full BPTT, spectral cap),
which is the only thing that has ever earned depth. Does keeping HALF the steps on
that recipe preserve the depth curve while the db1 half buys wall-clock? Risk: the
two halves train different functions through shared weights (the db1 half trains a
K-hostile one-pass map — dbfix proved that) and may interfere.

l2cap_cond: does per-iteration AdaLN conditioning HELP a loop that already works?
(Zero-init ⇒ starts bit-identical to l2cap; the question is what the learned
modulation does to the depth curve and CE.)

## Reference numbers (fixed)

l2cap: CE@4250 4.3489, depth-earned 0.233 nats (4.6220@K1 → 4.3892@K6), 66 min,
greedy rep4 0.61. dbfix: CE@4250 4.4521, ladder INVERTED, 37.7 min. db_cond:
CE@4250 4.3584, depth flat. Coreless bar 4.3102, ~30 min. Replicate CE spread
0.030–0.036. All: batch 6, 4500 steps, seed 1, eager, panel flags.

## Predictions (frozen)

- **I1 (ilv50 stability).** S1-clean: 85% (both halves individually stable; the
  mixed cycle itself is the only new surface).
- **I2 (ilv50, binding).** Depth sweep CE(K=1) − CE(K=6) ≥ 0.10 nats (≥43% of
  l2cap's 0.233): 40%. My prior leans fail — the db1 half trains a map the Euler
  ladder makes worse, and gradient interference through shared weights can poison
  the bptt half's contractive solution.
- **I3 (ilv50 wall-clock).** ≤ 55 min (guesstimate 51±5): 70%.
- **I4 (ilv50 CE).** CE@4250 ≤ 4.40: 50%.
- **LC1 (l2cap_cond stability).** S1-clean: 90%.
- **LC2 (l2cap_cond, binding).** Depth-earned ≥ 0.20 nats (retains l2cap's curve
  within noise): 55%. Zero-init protects the start; the risk is the conditioning
  giving iterations an easy identity shortcut.
- **LC3 (l2cap_cond CE).** CE@4250 < 4.3489 (beats l2cap): 35%.
- **Decision rules (binding).** I2 passes ⇒ step_mix is a real wall-clock lever:
  next arm is a mix-ratio probe (75/25) OR fold step_mix into the capped default —
  Wolfe picks. I2 fails ⇒ the interleave line is closed permanently (one-pass
  steps dilute or poison the loop; wall-clock must come from compile/kernels
  instead). LC2 AND LC3 pass ⇒ conditioning joins the gate-vs-cap ladder as a
  fourth arm (cap+gate+cond combinations). LC2 fails ⇒ conditioning stays out of
  capped recipes for good. S1 as before: no eval >0.20 nats above running min for
  2+ consecutive evals after step 1000.

## Method

1. Smoke ilv50 first (steps=12, eval_every=5): the MIXED step cycle has never run
   live (dbfix ran pure db1) — gate on exit 0 + startup line showing
   `cycle=['bptt','db1']` (or equivalent order) + spectral projection armed
   (`Core spectral PROJECTION ON: cap=1.5`). Smoke l2cap_cond: gate on exit 0 +
   projection armed + conditioning constructed (`core_stage_cond=iter` in the
   model banner or config echo).
2. Run tul-ilv50 then tul-l2cap-cond, panel flags, seed 1; each followed by the
   48-row core_depth_sweep (l2cap_cond uses forced-depth mode, NOT the Euler
   ladder — it has no sigma path; ilv50 gets BOTH sweep modes: forced-depth for
   the bptt-trained loop and db1_ladder_steps for the sigma path) and
   tul_samples; checkpoints pruned to step_4500.
3. Wall-clock from queue-log START→DONE stamps. Artifacts →
   lab/experiments/results/2026-08-30-tul-ilv50-l2capcond/.

## Not verified before launch

The mixed bptt+db1 cycle has never executed in a live main() loop. The
iter-conditioning + spectral-projection combination has never run on GPU (CPU
bit-identity only). The dual-mode sweep for ilv50 (forced-depth AND ladder on one
checkpoint) has not been exercised.
