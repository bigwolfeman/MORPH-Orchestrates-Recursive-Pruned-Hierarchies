# Planned: bound-superposition seed — static rank check

Status: failure
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

### Method amendment — 2026-09-01 09:35 (before any data read)

The probe's assert surfaced that `tul_g0c0` (the 20k arm's config, via the
`tul_l2` line) trains with `slot_seed: boundary` — E_slot + W_sent·embed of
the span's LAST token — not the spec-default bag_mean. So the shipped 20k
slots carry at most one token's identity. The probe now measures THREE seed
populations on the same spans: `ship` (the arm's actual boundary seed, via
`slot_input`), `bag` (E_slot + span mean, the spec default the predictions
name), and `bound`. P-B1/P-B2 score exactly as frozen (bag vs bound); `ship`
is a diagnostic column. Predictions untouched.

## Results — 2026-09-01 09:33 (201 rows, exit 0)

Shipped mode confirmed `boundary` (Method amendment above). Artifact:
/home/wolfe/morph-scratch/tulfm/bound_seed_rank.json.

| population | rank_unit | rank_raw | mean pairwise cos |
|---|---|---|---|
| ship (boundary, the arm's seed) | 1.58 | 1.30 | 0.773 |
| bag (spec default) | 1.13 | 1.14 | 0.941 |
| bound | 3.07 | 3.07 | 0.558 |
| ship − E_slot | 2.59 | 1.43 | 0.524 |
| bag − E_slot | 3.47 | 3.84 | 0.507 |
| bound − E_slot | **38.30** | **38.07** | **0.061** |

- **P-B1 FALSE** (80% miss): bound/bag unit-rank ratio 2.73 < 10 on the
  with-E_slot populations the prediction named.
- **P-B2 FALSE** (60% miss): bound cos 0.558 ≥ 0.1.

## Verdict

Filed as a failure on the frozen metrics, and the diagnostic decomposition
the prereg carried explains WHY: **the binding mechanism works exactly as
theorized on the bare content term** (rank 38.3 vs 3.5, cos 0.061 vs 0.507 —
an 11x rank lift), **but the shared E_slot vector dominates every seed's
norm and re-collapses the population to ~rank-1** (bag with E_slot: 1.13).
This also explains two old findings at once: the trained slot states'
rank 1.7-4.8 / cos +0.39..+0.71 at every checkpoint, and TG4a's "the seed
moves CE by 0.003 nats" — the content term was negligible against E_slot in
every mode, so seed-mode changes could not matter. What the frozen method
could not distinguish: it scored the composite seed, and the composite is
dominated by a term the binding never touches.

## Updated hypothesis

The seed defect is TWO defects: an unbound (or single-token) content term,
AND E_slot scale dominance. E3's design updates per the prereg's own
contingency clause: bound content term + E_slot rebalancing (shrink the
E_slot coefficient or RMS-match the content term) so the content term
carries the population geometry. Next planned experiment: E3 with both
changes, rank gate re-measured on the TRAINED states.
