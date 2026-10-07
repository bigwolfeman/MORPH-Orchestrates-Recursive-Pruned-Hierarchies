# Agent Note: LXTUL + pointer + DITTO is the winning recipe

Status: implemented

## Problem

The locked LXTUL candidate ([2026-10-04 note](2026-10-04-lxtul-primary-candidate.md)) earns
loop depth (K1-K6 +0.0215 / +0.0172 at 5k), but it sits 0.272 nats behind a plain model at
5k, and 86 % of that gap is far-bigram tokens: the strict geometry has no cheap way to copy a
token from an earlier span. A copy channel closes the gap but makes sampled text repeat more.

## Decision

Wolfe declared the winner on 2026-10-07. The recipe is two stages:

1. `morph/configs/lxtul_pointer.yaml`, 5000 steps: `lxtul.yaml` plus `tul.pointer_heads: 4`,
   the output-only pointer head (`morph/model/tul_pointer.py`). Each head attends from a
   token's final hidden state over earlier tokens and reads the attention as a distribution
   over the token that followed. A null key lets a head abstain. Nothing enters a hidden
   state, so the loop stays the only cross-span channel inside the model.
2. `morph/configs/lxtul_pointer_ditto.yaml`, a 1000-step phase from the stage-1 checkpoint:
   `tul.ditto_rows: 3` of each batch's 6 rows are pseudo-repetition rows, scored with the
   DITTO loss (Xu et al. 2022) at lambda 0.5 on the pointer-mixed probability
   (`morph/model/tul_ditto.py`).

Evidence, MORPH 5090, one seed per arm:

| model | CE@6, 480 rows | gap to plain (no pointer) | K1-K6 | T 0.7 seq_rep_4 |
| --- | --- | --- | --- | --- |
| plain | 4.0806 | 0 | - | 0.215 |
| LXTUL | 4.3529 | +0.272 | +0.0215 | 0.041 (degenerate text) |
| LXTUL + pointer | 3.9280 | -0.150 [-0.169, -0.132] | +0.0155 | 0.307 |
| + DITTO phase | trainer val +0.071 vs its control phase | not measured on fresh rows | +0.0294 | 0.140 |

The head-off check is clean: with the head forced off, CE is 4.41, which is 0.05 worse than
an LXTUL that never had a head. Filings:
[pointer pair](../../../../lab/experiments/mixed/2026-10-06-lxtul-pointer.md),
[coverage phase](../../../../lab/experiments/failures/2026-10-07-lxtul-pointer-coverage.md),
[DITTO phase](../../../../lab/experiments/mixed/2026-10-07-lxtul-pointer-ditto.md).

## Alternatives considered

- **Exact copy cache** (Parcae testbed): closed the gap to +0.022, but Wolfe called it a hack
  that a final model should not carry. The learned head beat it by 0.095 of head worth.
- **Pointer cell keys** (`tul.pointer_cell_key`): -0.0073 paired, inside the 0.009 seed
  noise, K1-K6 fell to +0.0098, one recovered detonation. Not taken.
- **See et al. coverage** as the repetition fix: measured negative, sampled repetition rose
  29 %. Coverage counts attention per position; an LM loop copies from fresh positions.
- **No repetition fix**: the pointer model repeats 1.4x as much as plain at T 0.7.

## Consequences

- New loop, CE and speed work starts from `lxtul_pointer.yaml` (+ the DITTO phase).
- Open and measured: DITTO costs +0.071 trainer val against its control phase; the phase
  introduced junk subwords in samples (not yet counted).
- Owed: a MORPH plain + pointer ruler (on Parcae, plain + pointer led strict + pointer by
  +0.045); a seed twin of the pointer run; a DITTO-from-start or lighter-DITTO variant.
- Throughput: 11.8k tok/s against Parcae's 23.5k on the same rows. The profiling round of
  2026-10-07 owns that gap.
