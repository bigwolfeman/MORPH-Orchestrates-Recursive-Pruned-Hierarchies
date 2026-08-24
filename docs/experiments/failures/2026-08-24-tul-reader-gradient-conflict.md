# Experiment: do the slot's readers pull it apart?

Status: failure

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

---

# Results

Ran 2026-08-24 on the 11-rung onset ladder, batch 4, 16 slots per rung, up to 12 readers
per slot (mean K = 9.6). `lab/divergence/reader_conflict_probe.py`.

| step | core share | K | alignment | conflict | mean pair cos | route_frac | cos(direct, readers) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1625 | 0.017 | 9.6 | 1.415 | 0.4730 | +0.0067 | 0.4539 | +0.014 |
| 1650 | 0.018 | 9.6 | 1.398 | 0.4724 | +0.0157 | 0.4581 | −0.005 |
| 1675 | 0.021 | 9.6 | 1.401 | 0.4720 | +0.0135 | 0.4725 | +0.007 |
| 1700 | 0.042 | 9.6 | 1.432 | 0.4834 | +0.0141 | 0.4747 | +0.012 |
| 1725 | 0.017 | 9.6 | 1.433 | 0.4822 | +0.0194 | 0.4782 | −0.004 |
| 1750 | 0.051 | 9.6 | 1.453 | 0.4895 | +0.0225 | 0.4507 | −0.000 |
| 1775 | 0.071 | 9.6 | 1.395 | 0.4748 | +0.0221 | 0.4836 | +0.002 |
| 1800 | 0.388 | 9.6 | 1.381 | 0.4690 | +0.0215 | 0.4832 | −0.025 |
| 1825 | 0.181 | 9.6 | 1.359 | 0.4615 | +0.0138 | 0.4972 | −0.020 |
| 1850 | 0.929 | 9.6 | 1.430 | 0.4849 | +0.0203 | 0.5183 | +0.023 |
| 1866 | 0.961 | 9.6 | 1.370 | 0.4675 | +0.0289 | 0.5409 | +0.007 |

A second pass with the reader cap raised to 40 (mean K = 16.3, 10 slots) to test the
"route_frac is an upper bound" caveat:

| step | K | alignment | conflict | mean pair cos | route_frac |
|---:|---:|---:|---:|---:|---:|
| 1700 | 16.3 | 1.259 | 0.3716 | +0.0085 | 0.5923 |
| 1866 | 16.3 | 1.216 | 0.3569 | +0.0211 | 0.6432 |

The two passes sample different slot sets (10 picks versus 16 from the same deterministic
spread), so their `route_frac` values are not directly comparable to each other. What is
comparable within each pass is the trend across rungs, and both passes agree on it.

## Prediction scorecard

**One of four holds, and it holds trivially.**

| # | prediction | result | measured |
|---|---|---|---|
| P1 | alignment at or below 1.3 at every rung | **FAILS** | 1.359 to 1.453 — readers agree BETTER than random at every rung |
| P2 | alignment falls at least 20 % from 1700 to 1866 | **FAILS** | 1.432 -> 1.370, a fall of 4.3 %, inside the rung-to-rung spread |
| P3 | route_frac below 0.25 at every rung | **FAILS** | 0.451 to 0.541, and 0.59 to 0.64 with more readers |
| P4 | cos(direct, readers) below 0.5 | **holds trivially** | −0.025 to +0.023 — the two routes are ORTHOGONAL, not opposed |

## Verdict: refuted

**The readers do not pull the plan apart.** Normalised alignment sits at 1.36 to 1.45, i.e.
about 40 % better than K independent random directions, with a mean pairwise cosine of
+0.02. They are mildly aligned. And it is FLAT: the spread across the onset (0.094) is
smaller than the spread between adjacent healthy rungs. Nothing about this geometry changes
while the core share goes from 0.017 to 0.961.

**The premise of the accounting was also wrong, and this is the more useful correction.**
The slot's own label carries 2.8 % of its span's LOSS WEIGHT but delivers about **half the
gradient** that reaches `h_i` — `route_frac` 0.45 to 0.64. Loss weight is not gradient
share. The direct route is short (one `W_prefix` projection into one coda position) while
the reader route passes through the coda's attention, which attenuates it. Every statement
of the form "the core is trained 97 % indirectly" was reasoning about the wrong quantity.

**And the two routes are orthogonal**, cosine within 0.025 of zero at every rung. The
slot's own label and its readers are not fighting; they are asking for different things in
different directions, and both get what they ask for.

## What survives, and what does not

Dead: the mechanism in
`.agents/notes/proposed/architecture/2026-08-24-xm-applies-to-the-plan-not-the-head.md`
— that `h_i` is a compromise between conflicting readers. The note moves to `rejected/`.

Untested, and it was flagged as the strongest counter-argument in that note before this ran:
under teacher forcing the next span is GIVEN, so this probe measures disagreement among the
readers of ONE known continuation. The Explorative Modeling claim is about one-to-many
coupling ACROSS the dataset — the same slot state serving different continuations in
different examples. This experiment cannot see that, and does not refute it. Testing it
needs the same context with different continuations, which is a data-construction problem,
not a probe.

## What this cannot decide

* Readers are the token positions of span `i+1`. Positions further along also read the slot
  through the coda's attention and are not counted.
* Batch 4, one seeded set of Poisson depths, one run's ladder, 16 slots per rung.
* The gradient is measured at the plan `h_i`, AFTER the loop. It says nothing about how
  those gradients are transformed by the 24 applications of `J_core^T` behind it.
* Correlation only, as with everything else in this campaign.
