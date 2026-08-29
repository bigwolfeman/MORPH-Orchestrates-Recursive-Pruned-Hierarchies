# Planned: Oracle prefix probe — can the coda decode ANY content planted at a slot prefix?

Status: planned
Date: 2026-08-28
Checkpoint: `checkpoints/morph/fm1-cw-s1/step_4500.pt` (run `hzwjin1e`)
Prior: `failures/2026-08-28-tul-fm1-cw.md` named this probe as the next step.

## Question

FM1-CW showed slot positions worth 0.81 nats while plan content is worth 0.0001.
Is that because the additive `W_prefix(z)` interface cannot deliver content the coda
can decode, or because the planner's z (though retrievally accurate) carries nothing
decodable? Plant PERFECT content — the true target y — and measure.

## Scope caveat (recorded before running, per Wolfe)

This probe is eval-only on a coda TRAINED to ignore z-content. A NEGATIVE result is
weakly conclusive (reading machinery may simply never have formed and could form
under training with useful content). A POSITIVE result is strongly conclusive (the
pathway can decode content it never saw — the fault is the planner's basis).
Wolfe's standing diagnosis: the coda attention-sinks on the latents and shortcuts
surface predictions; possibly a training-length effect. This probe cannot refute
that; it can only detect a decodable pathway if one exists.

## Method

`lab/tulfm/oracle_prefix_probe.py`, eval-only, 100 val batches (batch 6), paired
per batch. Six conditions on one fixed eval set:

1. `normal` — the shipped plans.
2. `zero`, 3. `shuffle` — the run's own reference ablations (must reproduce
   ~0.005 / ~0.0001 from the run, sanity anchor).
4. `oracle_y_unit` — replace each valid slot's plan with its true target y_i
   (unit-norm — 30× the plans' trained input scale; deliberately OOD in norm).
5. `oracle_y_scaled` — y_i rescaled to the mean norm of the real plans (content
   at the trained interface scale).
6. `zero_scaled_noise` — fresh N(0, mean_plan_norm²/d) vectors (control: is any
   effect of (5) mere norm/energy, not content).

Report per condition: `ce_tokens` mean over the fixed set, delta vs `normal`, and
a 95% bootstrap CI over batches for the two oracle deltas.

## Predictions (frozen)

- **H1.** Both oracle deltas are ≈ 0: |Δce| < 0.01 nats. (The coda's reading
  machinery does not exist; content at the interface changes nothing.) This is my
  honest expectation and matches Wolfe's diagnosis.
- **H2.** `oracle_y_unit` is, if anything, WORSE than normal (Δ > −0.005 not
  required; the 30× norm is OOD for W_prefix and may add noise).
- **H3.** `zero` and `shuffle` reproduce the run's readings within noise
  (zero ≈ +0.005, shuffle ≈ +0.000, both relative to normal).
- **H4.** `zero_scaled_noise` ≈ `zero` within 0.005 (energy alone does nothing).

## Decision rule

- Oracle (either scale) IMPROVES ce_tokens by ≥ 0.02 with CI excluding 0 ⇒ the
  pathway CAN decode content ⇒ the fault is the detached planner basis; next step
  is trainable alignment (un-detach a low-bandwidth projection, or train W_prefix
  toward the oracle).
- All oracle deltas < 0.01 ⇒ consistent with "no reading machinery formed";
  decision moves to the principled-reliance design (research synthesis in
  progress), NOT to more eval probes. Files to `failures/` (H1 predicted, so the
  filing is expected).
