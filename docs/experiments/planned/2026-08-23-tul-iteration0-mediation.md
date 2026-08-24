# Experiment: is the TUL takeover caused by the FIRST loop iteration alone?

Status: planned. Written BEFORE the M0/M1 runs, while the Phase 0.4 replicate pair is
still on the GPU.

## Question

[Phase 1](../results/2026-08-23-tul-onset-ordering.md) found that the looped core's
realized gain runs away **only on iteration 0**: `core_gain_t0` 1.422 → 16.93, `t1`
1.080 → 1.596, and `t2..t7` never leave their baseline, at every detection threshold.
`core_gain_max` equalled `core_gain_t0` at every probed step of the run.

That is an ORDERING and a LOCALISATION. It is not causation. This experiment asks the
causal question with an intervention that can only act in one place.

## Hypothesis

The takeover is driven by the runaway of the **first** core iteration's update. Iterations
1..T−1 are contracting what iteration 0 expands, and are not where the disease lives.

Corollary worth stating because it reframes a known result: `core_gain_clip=1.5` is one of
the four surviving cures, and it was always applied to EVERY iteration. Given the control's
measured per-iteration gains (t0 baseline 1.422; t1..t7 between 1.08 and 1.13), a cap of
1.5 can almost never bind at t ≥ 1. **The known cure may already be, in effect, "clip
iteration 0".** If so, three of the four "independent" cures may be less independent than
the plan assumes.

## Arms

Both compose `tul_a1` with `ademamix_alpha_cap=3.5` (the divergent control), seed 0, eval
and generation disabled, `grad_probe_every=1`, 5000 steps — identical to the replicate
pair except for the clip.

| arm | clip | iterations clipped |
|---|---|---|
| control | `core_gain_clip=0.0` | none (this is the replicate pair, already running) |
| **M0** | `core_gain_clip=1.5` | **t = 0 only** (`core_gain_clip_iter_lo=0, _hi=0`) |
| **M1** | `core_gain_clip=1.5` | **t ≥ 1** (`core_gain_clip_iter_lo=1, _hi=-1`) |

`core_gain_clip_iter_lo/_hi` do not exist yet. They are added in the same change, default
`(0, -1)` = every iteration = today's behaviour exactly. A `clip_bind_t` probe is added
with them: the fraction of samples whose scale < 1 at iteration t, which measures where a
clip actually acts rather than where it is nominally applied.

## Predictions

- **P1.** M0 does NOT take over by step 5000: pre-clip core share stays below 0.5
  (sustained 25 probed steps) for the whole run.
- **P2.** M1 DOES take over by step 5000, and its onset step lies within the spread the
  replicate pair measures for the control. (If the replicate pair does not itself replicate,
  "within the spread" is unreadable and P2 falls back to "M1 takes over at all".)
- **P3.** In the control, `delta_ratio_t0` = ‖h_new − h‖/‖h‖ at iteration 0 departs from its
  baseline while `delta_ratio_t7` does not. That places the runaway in what the core ADDS,
  not in a shrinking denominator. **Falsifier:** `in_norm_t0` collapses while `delta_ratio_t0`
  stays flat — then the gain result was an artifact of the carrier's scale and Finding 1 is
  withdrawn.
- **P4.** In M1, `clip_bind_t` at t ≥ 1 is under 5 % of samples for the whole pre-onset
  phase, i.e. the t ≥ 1 clip is nearly inert. This is the sharp version of the corollary:
  it says the known `core_gain_clip=1.5` cure was never really acting at t ≥ 1.

## What each outcome means

- **P1 and P2 both hold** → the disease is localised to iteration 0. The four "independent"
  cures need re-reading, and Phase 2's mediation shrinks to one candidate.
- **P1 holds, P2 fails** (both arms survive) → clipping anywhere cures it, the localisation
  is not causal, and the gain is a symptom. Finding 1 stays descriptive.
- **P1 fails, P2 holds** (both diverge) → iteration 0 is not sufficient to explain it, and
  the cure must act through something the per-iteration clip does not touch.
- **P1 and P2 both fail** → the arms are indistinguishable and the run-to-run noise measured
  by the replicate pair swamps the effect. Report as inconclusive under `failures/`, with
  the noise floor stated.

## Known limits, stated before the runs

- **n = 1 per arm.** The replicate pair is the only estimate of run-to-run spread, and it is
  itself n = 2. A single surviving arm is weak evidence; the plan's own §14 warns that the
  n=1 sweep measured trajectory sensitivity, not causation. This experiment is a
  LOCALISATION test with a strong prior from Phase 1, not a replacement for Phase 2.
- **"Survived" means 5000 steps**, roughly 2× the control's measured lifetime, not the 20k
  at which the known cures were validated. A 5000-step survivor is a first read.
- Changing the clip changes the forward pass, so M0 and M1 are not the same model as the
  control after step 0. Any loss comparison between arms is confounded and is not used.
