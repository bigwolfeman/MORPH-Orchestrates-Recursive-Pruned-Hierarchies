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

---

## Results (filed 2026-08-28, ckpt `fm1-cw-s1/step_4500.pt`, 100 val batches, paired)

Status: success (all four predictions held)

Artifact: `lab/experiments/results/2026-08-28-oracle-prefix-probe/probe.json`.
Script: `lab/tulfm/oracle_prefix_probe.py`.

| condition | ce_tokens | Δ vs normal | 95% CI |
|---|---|---|---|
| normal | 4.5791 | — | — |
| zero | 4.5847 | +0.0056 | [+0.0051, +0.0060] |
| shuffle | 4.5794 | +0.0003 | [+0.0002, +0.0005] |
| oracle_y_unit | 4.5791 | −0.0000 | [−0.0001, +0.0001] |
| oracle_y_scaled | 4.5791 | −0.0000 | [−0.0001, +0.0001] |
| zero_scaled_noise | 4.5794 | +0.0003 | [+0.0002, +0.0005] |

- H1 HELD: both oracle deltas ≈ 0 (|Δ| < 0.0001, far under the 0.01 line).
- H2 HELD (vacuously strong): unit-norm oracle did not even hurt — the coda is so
  content-blind that a 30× norm excursion at the prefix changes nothing.
- H3 HELD: zero/shuffle reproduce the run's readings.
- H4 HELD: scaled noise == shuffle exactly (+0.0003) — the entire non-zero part of
  the "content" signal is an is-something-there energy cue, not content.

Note: the prereg's decision-rule line said this outcome "files to failures/" —
that contradicted the protocol (success = the predictions held; they all did).
Filed under successes/ per protocol; the misprediction was in the label, not the
science.

## Verdict

The additive `W_prefix(z)` interface, as trained in FM1-CW, delivers zero
decodable content — even PERFECT content (the true target itself). Combined with
the scope caveat (eval-only; reading machinery may form under training with a
reason to form), the conclusion is exactly the pre-registered branch: no more
eval probes; the next move is the principled-reliance design. The strongest
single lead from the research sweep: the TG paper's own Table 1 ablation measured
that DETACHING the memory-write gradient costs 10× PPL — and FM1's z is fully
detached. The reading machinery has no gradient reason to exist and (He et al.
2019) the collapsed optimum is stable from initialization.
