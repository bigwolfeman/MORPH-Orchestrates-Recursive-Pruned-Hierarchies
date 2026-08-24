# Experiment: do the slot's readers pull it apart?

Status: planned

Committed before the probe was written. Acceptance criteria come from
[.agents/notes/proposed/architecture/2026-08-24-xm-applies-to-the-plan-not-the-head.md](../../../.agents/notes/proposed/architecture/2026-08-24-xm-applies-to-the-plan-not-the-head.md).

## Question

The TUL slot state `h_i` gets 2.8 % of its span's loss weight from its own label. The other
97 % arrives as a SUM of per-reader gradients through the coda's attention. If those readers
demand different things, the sum is a compromise and `h_i` lands near the centroid of all
spans — which is what the measured post-loop effective rank of 1.7 to 4.8 in 1024 dimensions
looks like.

Is the summed gradient a compromise, or do the readers agree?

## Method

For a slot `i` in row `b`, with readers `r` = the token positions of span `i+1`:

    g_r = d(CE at r) / d(h_i)          one autograd.grad per reader, retain_graph
    g_A = d(CE at the slot's own emitting position) / d(h_i)

    conflict   = || sum_r g_r || / sum_r || g_r ||
    alignment  = conflict * sqrt(K)
    route_frac = || g_A || / ( || g_A || + sum_r || g_r || )

`alignment` is the reported statistic because `conflict` alone is not comparable across
different reader counts: K independent random directions in high dimension give
`conflict ~ 1/sqrt(K)`, so `alignment ~ 1` is the no-agreement baseline, above 1 is
agreement and below 1 is active disagreement.

`h_slots` is captured by wrapping `TULSlots.prefix_project` for the duration of one forward
and restoring it in `finally`. Per-position CE is computed from the captured coda hidden
state times the tied head for that ONE position, so no `[B, L, V]` logit tensor is
materialised.

Checkpoints: the onset ladder `ROLL_step_1625` (core share 0.017) through
`TAKEOVER_step_1866` (0.961), which brackets the transition at 25-step spacing.

## Predictions

**P1.** `alignment` is at or below 1.3 at every rung — the readers are no better than
random agreement, so the summed gradient carries little more direction than noise.

**P2.** `alignment` FALLS across the onset: its value at 1866 is at least 20 % below its
value at 1700.

**P3.** `route_frac` is below 0.25 at every rung — the direct label is a minority of the
gradient reaching `h_i`, as it is a minority of the loss weight.

**P4.** `cos(g_A, sum_r g_r)` is below 0.5 at every rung: what the slot's own label wants is
not what its readers want.

**Refuted if** `alignment` is well above 1 and flat, which would mean the readers agree, the
summed gradient has a clear direction, and `h_i` is not a compromise. In that case this line
dies like the optimizer-state line and the under-determination line before it, and the note
above must be marked rejected.

## What this cannot decide

* Correlation. A conflicted gradient at onset does not prove that removing the conflict
  prevents the takeover; that needs one of the three levers built and run.
* Teacher forcing. Within one example the next span is given, so this measures conflict
  among readers of ONE known continuation, not one-to-many coupling across the dataset. A
  low alignment here is consistent with the XM reading but does not establish it.
* One batch, one seeded set of Poisson depths, one run's ladder.
* Readers are taken as the token positions of span `i+1`. The coda's attention lets LATER
  positions read the slot too; those are not counted, so `sum_r ||g_r||` is a lower bound
  and `route_frac` an upper bound.
