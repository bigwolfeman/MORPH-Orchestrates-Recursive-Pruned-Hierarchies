# Planned: bound-superposition seed — static rank check

Status: planned
Date: 2026-09-01 (frozen before the probe runs; rung E2 of
.agents/notes/proposed/architecture/2026-09-01-slot-channel-recovery.md).

## Question

Does HRR-style binding (per-offset frozen orthogonal rotation before the
span sum) fix the slot-seed rank collapse by construction, on real spans and
the real trained embedding table?

## Hypothesis

The bag-mean concentrates (measured slot-state effective rank 1.7-4.8,
pairwise cos +0.39..+0.71); binding decorrelates summands, so bound seeds
approach the rank of the span population itself. Near-theorem — the check
guards the implementation, not the math.

## Method

No training. Load the tul-20k checkpoint's embedding path on eval rows
(the same signal `bag_mean` pools: the per-position embedding entering the
slot seed). For each packed row, compute per span:
`s_bag = E_slot + mean_k(e_k)` (today's seed) and
`s_bound = E_slot + (1/sqrt(N)) * sum_k R_k @ e_k` with R_k fixed random
orthogonal (QR of a seeded Gaussian, one per within-span offset, frozen).
Per row, over its valid slots: effective rank = participation ratio of
squared singular values (the jac_ladder convention, raw and unit-normalized)
and mean pairwise cosine. Report means over >= 200 rows, both seed types,
plus the no-E_slot variants as diagnostics.

## Predictions (frozen)

- **P-B1.** Per-row effective rank (unit-normalized) of bound seeds >= 10x
  bag-mean seeds': 80%.
- **P-B2.** Mean pairwise |cos| of bound seeds (with E_slot) < 0.1: 60% —
  lower than P-B1 because the shared additive E_slot term inflates cosine
  for both seed types; the no-E_slot diagnostic isolates it.
- **Binding.** Both TRUE ⇒ E3 (trained bound-seed arm) launches as
  designed. P-B1 FALSE ⇒ implementation bug — fix before ANY training; the
  theory does not fail quietly here. P-B1 TRUE / P-B2 FALSE ⇒ E3 adds
  E_slot-scale rebalancing (e.g. smaller E_slot coefficient) to its design
  before launch.

## Not verified before run

Which exact signal the shipped bag-mean pools (embedding alone vs
embedding+bigram+value-embed) — the probe must read the call site in
`transformer.py` and pool the SAME signal; the prereg binds to "the same
signal", not to a named tensor.
