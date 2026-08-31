# Planned: future-leak vs past-memory — attributing the old l2cap depth-earning

Status: success
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

### Method amendment — 2026-08-31 11:05 (before the probe ran)

Added tul-30k/step_30000 (acausal override) as a third checkpoint: its
corrupt-K6 cell measures whether ANY honest multi-iteration capability formed
underneath the 3.85-nat leak. Also noting a third discriminator already in
flight at zero cost: the l2nc gen samples' repetition metrics vs the old
l2cap's greedy rep4 0.61 — at decode the carry is causal (all context is
past), so the old model's generation advantage, if it survives comparison,
is H-memory evidence independent of this probe.

### Method amendment — 2026-08-31 11:20 (v1 design flaw, discovered on read-out)

v1 corrupted only positions > k while scoring all token positions < k — but a
scored position's LABEL sits inside the clean region and inside the carry's
summary, so the label-leak path was intact for every scored position. v1 is
kept as a far-future result only: the 30k model's CE 1.2 persists with the far
future destroyed (its cheat is NEAR-future/label reading through the carry),
and the l2nc control is bit-identical clean-vs-corrupt (live causality proof,
P2 scored TRUE). v2 (boundary mode) corrupts token positions ≥ k and scores
ONLY the last token position before k — its label's input copy is corrupted —
sweeping k in a stride-64 grid. P1 is scored on v2. Predictions untouched.

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

## Results (v2 boundary mode, 16-point k-grid 128..1088 stride 64, 48 rows/k)

Aggregated over the k-grid (`$Q/future_leak_probe_v2.json`; per-k cells in the
file; re-measured from the artifact at filing time):

| arm | earning clean (K1−K6) | earning corrupt | collapse |
|---|---|---|---|
| l2cap-4500 (acausal_final) | **+0.2580** | **−0.0128** | **105%** |
| tul-30k (acausal_final) | **+3.2727** | **−5.5200** | 269% (K6 corrupt CE 9.84 — transcribes the garbage) |
| l2nc-4500 (carry none) | +0.0171 | +0.0171 | 0% — bit-identical clean vs corrupt |

Sanity gate held: l2cap clean K1/K6 across the grid bracket the audit sweep's
4.622/4.389 (per-k spread is doc-content noise; the paired earning is the
statistic). v1 (far-future mode) is retained in `$Q/future_leak_probe.json` as
the far-future null: earning survives when only positions > k+lookahead are
corrupted, because the cheat reads the NEAR future / label copies.

## Verdict

**P1 TRUE** (≥70% collapse required; measured 105% at 4500, 269% at 30k) —
**H-leak. The l2cap depth-earning was 100% future-reading at teacher-forced
eval.** Corrupt-K6 going ABOVE corrupt-K1 (negative earning) means iterations
actively transcribe the corrupted future into the scored position.
**P2 TRUE** (l2nc moved 0.000000 < 0.01 nats at every depth — live causality
contract on a trained model).

Binding applied: the falsification stands as filed; the loop program stays
paused pending the loop-killer bisect (prereg
`lab/experiments/planned/2026-08-31-loop-killer-bisect.md`).

## Updated hypothesis

The carry's teacher-forced value is pure leak. Its GENERATION value (old
l2cap greedy rep4 0.614 vs l2nc 0.902) is not addressed by this probe and
remains real H-memory evidence at decode time, where the carry is causal by
construction — that motivates a future causal-carry design, but as a NEW
mechanism, not a rehabilitation of the old numbers.
