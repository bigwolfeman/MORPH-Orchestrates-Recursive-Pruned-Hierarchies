# Planned: TUL vs no-TUL at 20k on the loop-killer winner recipe

Status: success
Date: 2026-08-31 (frozen before launch; Wolfe: "20k run then... Lets get this
up and running". The honest re-run of the original 30k ask, shortened by
decision, on the campaign winner recipe.)

## Question

With the leak fixed and GLA+cap removed (the campaign winner: retention off,
spectral_project_cap 0, carry none), does TUL earn its keep at scale — on the
token axis, the wall-clock axis, or neither?

## Method

Two arms, sequential on the 5090, same panel flags (batch 6, seed 1,
alpha_cap 3.5, t_beta3 3500, eval_every 250, gen_every 0, grad_probe_every 1),
training.steps=20000, ckpt_every=2000 with post-run prune to step_20000.
- **A: tul-20k** — `tul_g0c0` (tul_l2 + retention=false + cap=0; TUL from
  step 0, TG-restrict, eager kernels by requirement).
- **B: notul-20k** — `notul_bg0c0` (fused kernels), wandb.name notul-20k.
Readouts: val CE trajectory (val/loss), wall-clock per arm from the queue-log
epochs, `core_depth_sweep.py` (A) / `token_depth_sweep.py` (B) at 1..8,
48 rows, gen samples with rep4/distinct-3. Wall-clock-matched comparison:
A's final CE vs B's CE at the step B reaches A's total wall time.
Stability risk accepted: the no-cap recipe is untested past 4500 steps;
div-guard is the instrument, a detonation is itself a result (contraction
prerequisite evidence).

## Predictions (frozen)

- **P1 (token axis).** B's final val CE < A's at matched 20k steps: 75%
  (B led by 0.096 at 4500; some chance TUL amortizes).
- **P2 (wall clock).** A's total wall time ≤ 0.85 × B's: 80%.
- **P3 (stability).** BOTH arms reach 20k without div-guard abort: 60%
  (two exposures of an uncapped core to a 4.4× longer horizon).
- **P4 (the value prop).** At matched WALL CLOCK, A's CE < B's: 45%.
- **Binding.** P4 TRUE ⇒ TUL ships as a supported first-class mode in the
  master merge and the cloud recipe gets a TUL variant. P4 FALSE ∧ P1 TRUE ⇒
  TUL stays merged but default-off (status quo), revisit only with R3-style
  span memory. Either arm detonates ⇒ contraction redesign becomes a merge
  BLOCKER for the no-cap recipe and the surviving comparisons are reported
  as-is.

## Not verified before run

No-cap stability beyond 4500 steps (either geometry); TUL never trained with
retention absent (smoke gates construction); disk: ~10 ckpts/arm × ~3.3 GB
transient before prune.

## Results — 2026-09-01 (scored 07:45, all four arms of the queue exit=0)

Runner `$Q/run_h2h20k.sh` (Q=/home/wolfe/morph-scratch/tulfm), launched
2026-08-31 22:59. Smokes passed 23:00:55 / 23:01:53. Queue-log epochs:

| Arm | Steps | Wall | Rate (steady) | TFLOPS | Final val CE | Last-5 mean | Best |
|---|---|---|---|---|---|---|---|
| tul-20k | 20000, exit 0, 79/79 evals | **4.504 h** | 1.47 sps | 11.2 | 3.8484 | **3.8461** | 3.6068@16250 |
| notul-20k | 20000, exit 0, 79/79 evals | **3.534 h** | 1.63 sps | 51.9 | 3.6195 | **3.4894** | 3.2736@16250 |

- **P1 TRUE** (75% side): notul wins the token axis by 0.357 nats on the
  last-5 mean (3.489 vs 3.846). The 4500-step gap (0.096) widened 3.7x with
  training; TUL did not amortize.
- **P2 FALSE** (80% side MISSED): wall ratio A/B = 1.275, vs the predicted
  <= 0.85. Both arms ran at steady state end to end (TUL tok/s 8956@7800 ->
  9051@19800; notul 9902 -> 10051), so this is not intra-run degradation:
  the prediction was anchored on a bad 4500-step timing memory (41 min) that
  does not describe this config. Root cause of the gap is the kernel path,
  not the architecture: TG-restrict forces use_kernels=false, and the eager
  path runs at 11.2 TFLOPS vs fused 51.9 — a 4.6x kernel-efficiency gap
  that TUL's ~2.3x FLOP reduction (proxy 10.2 vs 44.0) only half covers.
- **P3 TRUE** (60% side): both arms survived 20k uncapped, no div-guard.
  The winner recipe is stable at 4.4x the previously tested horizon in both
  geometries. Contraction redesign stays a Raven-gate, NOT a merge blocker.
- **P4 FALSE** (majority 55% side held): notul finished all 20k steps in
  less wall time than TUL took, so the matched-wall comparison degenerates
  to final-vs-final; notul wins by 0.357 nats.

Depth sweeps at step 20000 (48 rows):
- notul-20k token axis: K1 3.7161 -> K6 3.5089, K1-K6 = **0.207** (0.220 at
  4500 — the BG0C0 depth-earning replicates at 20k; still saturates by K4;
  K3-K6 = 0.016).
- tul-20k core axis: K1 3.7721 -> K6 3.7568, K1-K6 = **0.015**, saturated
  by depth 2. The winner recipe did NOT unlock loop earning for the TUL
  geometry; the ~0.017-nat core-loop value replicates at 20k.

Gen samples (vs real-text rep4 0.037 / distinct3 0.928): both arms healthy
at t=1.0 sampling (tul 0.026/0.960; notul 0.007/0.978), both collapse under
greedy (0.88 / 0.96 rep4) — normal at this scale. Artifacts:
$Q/{core,token}_depth_sweep_*.json, /home/wolfe/morph-scratch/gensamples/
{tul,notul}-20k_samples.json, checkpoints/morph/{tul,notul}-20k/step_20000.pt.

TUL train-loss lines read ~7.0 while val CE is ~3.85: the TUL training
objective is multi-term (token CE + slot terms); not an anomaly.

## Verdict

**Binding applied as frozen: P4 FALSE ∧ P1 TRUE ⇒ TUL stays merged but
default-off in the master ship.** No first-class promotion, no cloud TUL
variant. Revisit only with R3-style span memory
(.agents/notes/proposed/feature/2026-08-31-raven-memory-arms.md).

Caveat recorded for the revisit: P2/P4 are confounded by the eager-kernel
requirement. A fused TG-restrict kernel path (~4.6x headroom) could flip the
wall-clock axis without touching the architecture — but the token axis
(P1, 0.357 nats and widening) is kernel-independent and TUL loses it alone.

## Updated hypothesis

TUL as built is a FLOP-reduction mechanism whose savings the eager path
cannot cash, and its conditional-compute value does not survive the honest
(leak-fixed, GLA-free, uncapped) recipe at 20k. The loop earns ~0.02 nats in
the TUL geometry regardless of recipe — consistent with the slot-plan
finding (the plan carries ~0.07 nats). Span-level memory that WRITES
(R3/Raven slots), not span-level looping, is the live hypothesis for making
slot positions matter.
