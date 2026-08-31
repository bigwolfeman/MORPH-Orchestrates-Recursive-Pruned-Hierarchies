# Planned: future-leak vs past-memory — attributing the old l2cap depth-earning

Status: planned
Date: 2026-08-31 (frozen before the probe runs; trigger: Wolfe — "we had such
strong results before. something is wrong here... figure out the root cause").

## Question

The carry-off audit proved the l2cap-4500 depth-earning REQUIRES the carry. It
did not prove WHICH half of the carry: the state summarizes future AND past,
and it is also the loop's only cross-iteration memory. Both prior measurements
(carry-off eval; carry-none training) amputate both halves at once. H-leak:
the 0.233 was future-reading. H-memory: it was legitimate past-side
cross-iteration memory — in which case the loop really composed, and the fix
direction becomes a CAUSAL carry, not carry-none.

## Method

`lab/divergence/future_leak_probe.py`, checkpoints tul-l2-cap/step_4500
(carry ON as trained — explicit `model.retention_carry=acausal_final`, since
the post-fix default is "none") and tul-l2nc/step_4500 (causal control).
48 packed rows, batch 3. For k in {700, 900} (packed index of 1152): corrupt
INPUT ids at token positions with index > k (random non-special ids; slot
placeholder ids untouched — slot INPUTS are span bag-means over the corrupted
tokens, which is the point); labels untouched. Score CE only at token
positions with index < k (their labels are ≤ k, clean). Forced depths {1, 3, 6}
via the slot_mean_depth mutation. Earning(condition) = CE@K1 − CE@K6 over the
SAME scored positions; paired rows across all cells.

The window/CSA/HCA branches are causal at scored positions, so the ONLY path
from the corrupted region to a scored position is the acausal carry (and any
unknown leak — which this probe would also expose).

## Predictions (frozen)

- **P1 (attribution, binding).** l2cap-4500's earning at scored positions
  collapses ≥ 70% under corruption (earning_corrupt ≤ 0.3 × earning_clean),
  at both k: **60%** — H-leak favored (the 30k endpoint is pure cheat and 4500
  is on the same trajectory), but the old model's uniquely
  repetition-resistant GENERATION (no future available) is real H-memory
  evidence, hence only 60.
- **P2 (causal control).** tul-l2nc CE at scored positions moves < 0.01 nats
  under corruption at every depth: 90% — the causality contract, live, on a
  trained model.
- **Binding.** P1 TRUE ⇒ H-leak: the falsification stands as filed; the loop
  program stays paused. P1 FALSE (earning survives corruption) ⇒ H-memory:
  the 2026-08-30 recipe correction gets amended — the loop DID compose through
  legitimate past-side carry memory; the causal chunk-boundary carry gets
  designed and the recipe re-run with it; the l2nc "0.006" is re-read as
  amputation, not absence. Partial collapse (30-70%) ⇒ both mechanisms live;
  report the split and the causal-carry work item still opens.

## Not verified before run

Whether the acausal_final opt-in reproduces the pre-fix forward bit-for-bit on
a real checkpoint (the fork agent smoke-tested tiny models only) — the probe's
clean-condition K1/K6 numbers must match the audit sweep (4.622/4.389 ± noise)
as its own sanity gate, and a mismatch aborts the read.
