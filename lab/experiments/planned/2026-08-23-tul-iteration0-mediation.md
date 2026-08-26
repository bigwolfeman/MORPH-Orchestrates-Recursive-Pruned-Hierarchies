# Experiment: is the TUL takeover caused by the FIRST loop iteration alone?

Status: **planned, arms NOT run.** M0 was started and killed at step ~90; M1 never started.
See "Why the arms were stopped" below. The predictions stand and the experiment is ready to
run on the reproducible configuration.

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
and generation disabled, `grad_probe_every=1`, **4000 steps** — identical to the replicate
pair except for the clip.

*Corrected before the runs (2026-08-23): this said 5000 in the first draft, which
contradicted "identical to the replicate pair" — the pair runs 4000. All four arms now
share one step budget, which is the stronger design. Every "by step 5000" below reads
"by step 4000".*

| arm | clip | iterations clipped |
|---|---|---|
| control | `core_gain_clip=0.0` | none (this is the replicate pair, already running) |
| **M0** | `core_gain_clip=1.5` | **t = 0 only** (`core_gain_clip_iter_lo=0, _hi=0`) |
| **M1** | `core_gain_clip=1.05` | **t ≥ 1** (`core_gain_clip_iter_lo=1, _hi=-1`) |

**Amended before the runs (2026-08-23): M1's cap is 1.05, not 1.5.** The first draft used
1.5 in both arms for symmetry. That is a broken test. The control's measured gain at
t = 1..7 is 1.08–1.13, so a 1.5 cap there can essentially never bind, and M1 would have
been a no-op wearing a cure's clothes — "a no-op diverges" tests nothing. 1.05 sits BELOW
the measured t ≥ 1 gains, so M1 constrains iterations 1..T−1 aggressively and genuinely.
The arms are therefore no longer "the same cure in two places"; they are two different
questions, and that is the point:

- **M0** asks: is constraining iteration 0 ALONE enough to cure it?
- **M1** asks: is constraining everything EXCEPT iteration 0, and hard, still not enough?

If M0 survives and M1 dies, iteration 0 is both sufficient and necessary in this design.

`core_gain_clip_iter_lo/_hi` do not exist yet. They are added in the same change, default
`(0, -1)` = every iteration = today's behaviour exactly. A `clip_bind_t` probe is added
with them: the fraction of samples whose scale < 1 at iteration t, which measures where a
clip actually acts rather than where it is nominally applied.

## A structural fact found while the replicate pair was running

`_tul_core` seeds the loop with `h = gather(input_norm(prelude))` and then feeds each
iteration's output straight back in. **The loop has exactly one normalisation point, at
its entrance.** Measured on replicate A: `in_norm_t0` sits at 512.5 — which is exactly
`sqrt(64 slots × 1024 channels × 4 HC streams)`, i.e. unit RMS per element — and is
bounded above by that value at every step, while `in_norm_t7` floats between 457 and 933.

This gives Finding 1 a second reading that must be tested rather than assumed:

- **(a) the disease reading.** The map genuinely runs away on its first application.
- **(b) the visibility reading.** `gain_t` is a RATIO. At t = 0 the denominator is pinned
  by RMSNorm and cannot grow, so any output growth shows up in full. At t ≥ 1 the
  denominator is the previous iteration's output, so numerator and denominator grow
  together and the ratio is structurally damped. On that reading iteration 0 is not where
  the disease is — it is the only place the diagnostic can SEE it.

P3 discriminates them, because `delta_ratio` at t ≥ 1 is NOT damped the same way (measured
0.82–0.94 at t = 7 against 1.81–1.95 at t = 0 — the same order, not an order apart). If
`delta_ratio_t7` departs during the onset, reading (b) is live and Finding 1 must be
restated. If only `delta_ratio_t0` departs, reading (a) survives.

## Predictions

- **P1.** M0 does NOT take over by step 4000: pre-clip core share stays below 0.5
  (sustained 25 probed steps) for the whole run.
- **P2.** M1 DOES take over by step 4000, and its onset step lies within the spread the
  replicate pair measures for the control. (If the replicate pair does not itself replicate,
  "within the spread" is unreadable and P2 falls back to "M1 takes over at all".)
- **P3.** In the control, `delta_ratio_t0` = ‖h_new − h‖/‖h‖ at iteration 0 departs from its
  baseline while `delta_ratio_t7` does not. That places the runaway in what the core ADDS,
  not in a shrinking denominator. **Falsifier:** `in_norm_t0` collapses while `delta_ratio_t0`
  stays flat — then the gain result was an artifact of the carrier's scale and Finding 1 is
  withdrawn.
- **P4.** The known `core_gain_clip=1.5` cure was never really acting at t ≥ 1. Since M1
  now uses 1.05, this is tested OFFLINE and costs no GPU: on the control replicate runs,
  which carry no clip at all, the fraction of probed steps whose measured `core_gain_t` is
  at or above 1.5 will be under 5 % for every t ≥ 1, and large for t = 0. That is the
  counterfactual "would a 1.5 cap have bound here", read straight off the unclipped
  control, which is a cleaner measurement than an extra arm would have given.

## Amendment 2026-08-23, written while replicate A was finishing and BEFORE M0/M1 ran

**Replicate A did not take over.** At step 3600 its highest core share of the whole run is
0.1105, and that is a step-122 startup transient; the running value is 0.0143. The Phase 1
control took over at step 1985. Two things follow and both change what P1 and P2 can mean.

1. The divergence-by-N is **stochastic with a wide onset distribution**. RCA §13 already
   recorded control abort steps of 2080, 3240, 4540, 5900 and 6200 — a run reaching 4000
   healthy sits inside that range, it is not a surprise. With n = 1 per mediation arm and a
   control base rate somewhere near a coin flip at 4000 steps, **"M0 survived 4000 steps"
   carries almost no information.** P1 as written is close to untestable.
2. The step-budget confound above means the Phase 1 control (5000) and this pair (4000) do
   not even share an optimizer schedule, so the base rate cannot be pooled across them.

**So the readout moves from the EVENT to the RAMP.** The takeover is a threshold crossing
of quantities that ramp slowly and with low noise long before it. Replicate A over 4000
steps: `core_gain_t0` 1.34 → 1.77, `in_norm_t7` 457 → 1742 (a 4× inflation of the carrier
by the loop's last iteration), `spec/sigma_max` 1.43 → 3.89. Measuring whether an
intervention bends those ramps is far more efficient than waiting for an event that fires
at a random step, and it works even when no arm diverges.

Added readouts, all measured over the identical 0–4000 window in all four arms:

- **R1.** `spec/sigma_max` at step 4000, and its slope over steps 1000–4000. This is a
  property of the WEIGHTS, not the activations, so it is not mechanically forced by either
  clip. This is the primary added readout.
- **R2.** `in_norm_t7` at step 4000 — how much the loop inflates its own carrier by the
  last iteration. Downstream of both clips but not the clipped quantity in either.
- **R3.** `preclip/core` and the core share at step 4000.
- **R4.** `delta_ratio_t0`.

**Do not read `core_gain_t0` as a result in M0, nor `core_gain_t≥1` in M1.** Those are the
quantities each arm clips by construction; reporting them as an effect would be circular.
They are reported only as a manipulation check — evidence the clip did what it was told.

Predicted ordering, written now: if iteration 0 is where the disease lives, M0 bends R1 and
R2 toward flat and M1 does not, with the two control replicates bracketing M1.

**Honest limit:** n = 1 per arm and n = 2 for the control, so a trend difference smaller
than the control pair's own spread is not readable. The control pair is what sets that
floor, which is the other reason it was run first.

## What each outcome means

- **P1 and P2 both hold** → the disease is localised to iteration 0, and it is localised
  in the strong sense: t = 0 alone suffices to cure, and t ≥ 1 does not suffice even under
  a cap below its own healthy gain. The four "independent"
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
- **"Survived" means 4000 steps**, roughly 2× the control's measured lifetime, not the 20k
  at which the known cures were validated. A 4000-step survivor is a first read.
- Changing the clip changes the forward pass, so M0 and M1 are not the same model as the
  control after step 0. Any loss comparison between arms is confounded and is not used.


---

## Why the arms were stopped (2026-08-23)

M0 was launched and killed after ~90 steps; M1 was never started. The reason is the
replicate pair that ran immediately before them, and it is a design problem, not a
scheduling one.

**The control replicates disagree with each other by more than any effect these arms could
show.** `repl-det-a` and `repl-det-b` are byte-identical runs at the same seed. One finished
healthy (final core share 0.0152) and the other took over (0.8131). On the ramp readouts
this amendment introduced:

| readout | replicate A | replicate B |
|---|---|---|
| `core_gain_t0`, step 200 → 4000 | 1.466 → 2.028 (**+0.56**) | 1.339 → 2.981 (**+1.64**) |
| `delta_ratio_t0` | 1.792 → 2.241 (**+0.45**) | 1.284 → 3.116 (**+1.83**) |
| `in_norm_t7` | 557 → 1441 (**+884**) | 536 → 1110 (**+575**) |
| `spec/sigma_max` at step 3900 | 4.41 | 3.77 |

The two controls differ by 3–4× on the ramps. The amendment above said in advance that "a
trend difference smaller than the control pair's own spread is not readable", and the spread
turned out to swamp everything. With a control that takes over in one run of two, a single
surviving M0 is a coin flip. Running the arms would have spent 70 minutes of GPU to produce
a number that could not be interpreted, so they were stopped.

**This is not a negative result about iteration 0.** It is a statement that the experiment
could not be run in that configuration. Prediction P4 was scored, offline and at zero GPU
cost, and it held: on the unclipped control a `core_gain_clip` of 1.5 would have bound on
40.9 % of pre-onset steps at t = 0 and **0.0 %** at t = 2..7, before or after onset, so the
known cure never acted on the later iterations.

**What unblocks it:** a bit-reproducible configuration now exists
([`the result`](../results/2026-08-23-morph-bit-reproducible.md)). Re-run these arms with
`training.deterministic=true`, `model.use_kernels=false` and the reduced batch, with the
control re-measured in the SAME configuration — its takeover base rate and ramp spread do
not transfer from the fast configuration. That is the first experiment the reproducible
mode should be spent on.


## Second attempt, also not readable (2026-08-23, later)

Re-run in the reproducible configuration after the deterministic mode landed. All three
arms, 1600 steps, seed 0, on the fast venv:

| arm | max core share | final | `gain_t0` (median, last 200) | block gain |
|---|---:|---:|---:|---:|
| control | 0.3409 | 0.0114 | 1.552 | 0.959 |
| M0 (clip t=0 only) | 0.7384 | 0.0249 | 1.470 | 1.093 |

**Neither took over**, so the arms are again uninterpretable: M0 "surviving" says nothing
when the control also survives. M1 was not run.

The numbers are recorded rather than read. Taken at face value M0 looks slightly WORSE than
the control on both share and block gain, which is the opposite of P1 — but with no
divergence in either arm and n = 1, that comparison has no power, and stating it as a
finding would be exactly the error this file has already made once.

The cause is a budget problem, not a design problem: the same control took over at step
1093 on the other venv and ran healthy past 1600 here, because the two torch builds are not
numerically identical (29 of 30 steps differ). The onset step for THIS build is unknown and
must be measured before the arms are worth running.

**The prerequisite is now running** (`onset-capture`): the control on this build, to 5000
steps, with a rolling pre-onset checkpoint buffer. When it aborts it will give both the
onset step for this build AND a pre-onset state to replay from. After that these arms
should be run as resumes from that ONE checkpoint rather than from step 0 — identical
starting state, identical data stream, one variable changed. See
[`the replay cookbook`](../../cookbook/replaying-the-core-takeover.md). That is a strictly
better design than what this file specifies, and it costs ~300 steps per arm instead of
1600.
