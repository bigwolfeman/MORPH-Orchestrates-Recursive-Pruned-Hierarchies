# Planned: ilv50 + l2cap_cond — the revived pair (Wolfe's call, new questions)

Status: success
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

## Results

Wall-clock from queue-log stamps. Both arms trained clean, seed 1, batch 6, eager.
Smoke gates verified live before launch (mixed cycle `['bptt', 'db1']` + spectral
projection cap=1.5 on 12 core linears, quoted from the smoke logs).

| cell | bar | measured | verdict |
|---|---|---|---|
| I1 ilv50 S1-clean | no eval >0.20 over running min ×2 after step 1000 | max excursion +0.133 (step 2750) | **PASS** |
| I2 ilv50 depth (binding) | force-loop CE(K=1) − CE(K=6) ≥ 0.10 | **−0.0005** (4.4732 → 4.4737, flat to 4 decimals) | **FAIL** |
| I3 ilv50 wall-clock | ≤ 55 min | 54.2 min (13:29:48→14:24:02) | **PASS** |
| I4 ilv50 CE | @4250 ≤ 4.40 | 4.4441 | **FAIL** |
| LC1 l2cap_cond S1-clean | same as I1 | max excursion +0.142 (step 2750) | **PASS** |
| LC2 l2cap_cond depth (binding) | ≥ 0.20 nats | **0.0127** (4.4292 → 4.4165) | **FAIL** |
| LC3 l2cap_cond CE | @4250 < 4.3489 | 4.3727 (worse by 0.024, inside replicate spread — "no better") | **FAIL** |

Supporting numbers:

- tul-ilv50: final val 4.4975@4500, wandb `4e2iqeoc`. Euler-ladder (auto) sweep:
  shallow bowl, best K=3 (4.4883 vs 4.5062 at K=1), spread 0.018 — inert.
  Force-loop sweep of the bptt-trained loop: 4.4732/4.4737/4.4737/4.4734/4.4736/
  4.4737/4.4735/4.4736 for K=1..8. Samples: topk50 rep4 0.031, sample_t1 0.0015,
  greedy rep4 0.816.
- tul-l2cap-cond: final val 4.4235@4500, CE@4250 4.3727, 68.6 min
  (14:51:37→16:00:15), wandb `orzhq1a1`. Depth sweep: 4.4292/4.4183/4.4163/
  4.4157/4.4159/4.4165/4.4166/4.4166 for K=1..8 — earned 0.0127 vs l2cap's 0.233
  on the same instrument. Samples: topk50 rep4 0.057, sample_t1 0.0025, greedy
  rep4 **0.844** — l2cap's greedy resistance (0.61) is gone with the depth curve,
  the campaign-wide "generation health tracks mechanism" pattern again.
- Artifacts: `lab/experiments/results/2026-08-30-tul-ilv50-l2capcond/` (three
  depth sweeps, two eval histories, two sample JSONs).

## Verdict

Predictions: I1, I3, LC1 held at their favored priors; I2 (40%) and LC3 (35%)
failed on the side my priors leaned; I4 was a coin (50%) and failed; **LC2 (55%)
is the one miss** — I leaned pass and it failed hard. 5 of 7 held → filed to
successes with the miss named.

Both binding rules fire on their fail branches: **the interleave line is closed
permanently** (wall-clock must come from compile/kernels), and **conditioning
stays out of capped recipes for good**.

The pair's real finding is the reinterpretation LC2 forces: ilv50's total erasure
looked like mixed-objective poisoning, but l2cap_cond — the exact l2cap recipe
with ONLY the AdaLN-Zero iteration signal added, full BPTT every step — collapsed
the depth curve by ~95% on its own. Per-iteration conditioning is by itself
sufficient to kill depth-earning. Mechanism (consistent, not independently
probed): the conditioning gives the network a cheap per-iteration scale/shift to
differentiate iterations, so the dynamics no longer need to build a composition —
the iteration index is absorbed by AdaLN instead of by the map. ilv50 remains
confounded (mix + conditioning); with both lines closed the confound is moot.

## Updated hypothesis

Depth-earning under the capped recipe is fragile to ANYTHING that lets the
iterations differentiate without composing: a one-pass objective on alternate
steps, or a per-iteration conditioning signal on every step. The gate-vs-cap
ladder must therefore run its arms with NO conditioning module, and the gated
blend (Gated Recurrent Transformers) needs scrutiny on exactly this axis — a
learned per-iteration gate is ALSO an iteration-indexed shortcut unless the gate
input is state-dependent rather than index-dependent. Test state-keyed gates,
never index-keyed ones. A cheap discriminating probe exists before any new run:
zero the conditioning weights on the l2cap-cond checkpoint and re-sweep — if the
curve stays dead, the weights themselves reorganized; if it partially revives,
the shortcut is load-bearing at eval time.
