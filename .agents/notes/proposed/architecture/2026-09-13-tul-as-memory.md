# Agent Note: price the slot as a memory, not as a loop

Status: proposed

## Problem

Every slot arm in the arc has been scored on loop CONTRIBUTION — the K-curve, the worth
profile, the per-pass cotangents — and the answer has come back the same twelve times: the
passes are worth 0.0005 to 0.0033 nats. That is a verdict on the LOOP. It has been read,
in conversation and in filings, as a verdict on the SLOT. The two are different mechanisms
and nothing separated them.

A slot is also a MEMORY: a fixed-size cell per span that a later token reads instead of
re-reading the span. A memory is not priced in nats. It is priced in nats per unit of
inference cost, and that table did not exist.

## Proposal

`lab/divergence/memory_cost_table.py` plus `docs/tul-as-memory.md`, which defines the cost
model and holds the table. Per arm, on the SAME paired rows:

* **CE** at the arm's trained loop depth, read from the per-token sweep artifact
  `core_depth_sweep.py` already writes. Arms are paired on `tok_index`, the STREAM index,
  so arms that cut the same stream into different rows still pair at the TOKEN level.
* **Inference cost per GENERATED token at decode** — `block_passes_per_token`,
  `visible_keys_per_token`, `kv_bytes_per_token` — arithmetic from the arm's own composed
  config plus ONE measurement on real packed rows. The visible-key count goes through the
  SHIPPED relation builders, never a re-derivation.
* **The fraction of the cross-span budget recovered**,
  `(CE_spanlocal − CE_arm) / (CE_spanlocal − CE_full)`.

The 2026-09-13 table, on the 491,520 stream positions all seven arms share: the slot arms
are the cheapest row — **10.59** block-passes per generated token and **64.4** visible keys
against the depth-1 plain control's 14.00 and 512.5, an 8.0x cut in context and KV bytes —
and they pay 0.385 nats for it. `budget-web-span`, the same corpus with NOTHING crossing a
span boundary, is **0.092 nats WORSE** than `slot-spandec-strict` at 4.2x the
block-passes. The cells carry real cross-span information; the K-curve was never a
statement about them.

## Alternatives considered

* **A wall-clock table.** What anyone actually wants. Rejected for now because it needs
  the GPU, which is held, and because wall clock mixes the mechanism with kernel
  availability (the slot core runs eager under a `tg_allow` relation; the plain core runs
  fused). `block_passes_per_token` is the mechanism; the doc says explicitly that it is
  not wall clock.
* **Scoring on each arm's own `[VAL]` CE.** Rejected: the arms were swept on different
  packings and different row cuts, and `sweep-must-match-trainer-val` is already a filed
  lesson from 2026-09-09. Pairing on `tok_index` is the fix and it is printed.
* **Charging the span decoder at decode.** Rejected and stated: it is a training-only
  scorer, `_ternary_exclude` on every leaf, not in the deployed forward.
* **Printing the recovered fraction anyway.** Rejected as the default. The budget pair
  runs `core_impl: parcae` while every slot arm runs the MORPH ternary core, so the
  denominator is a scale from a different model. The script REFUSES the column, prints the
  config difference it found, and names the queued matched control. The
  `--allow-core-mismatch` artifact exists so the numbers are available with the caveat
  attached rather than unavailable.

## Acceptance criteria

1. Every cell in the doc traces to a file or a run. **Met** — no number was typed from
   memory.
2. The visible-key count comes from the shipped relation builders. **Met.**
3. Arms pair at the token level and the intersection size is printed. **Met**: 491,520.
4. A core mismatch blocks the recovered-fraction column by default. **Met.**
5. The matched control `plain-span-local` runs and the column stands on its own. **Not
   met — the config does not exist yet.**

Prereg: `lab/experiments/planned/2026-09-13-arc-tul-as-memory.md`.

## Risks

* **`model.span_mask` currently REQUIRES `core_impl: parcae` and `use_kernels: false`**,
  so `plain-span-local` may not be buildable as specified. Resolve before queueing: either
  the requirement is real and the matched control has to come from the other direction
  (the slot family on a Parcae core), or it is a guard that can be widened with the
  existing leak test. Recorded as the first thing to check.
* **An unplanned cross-check went the right way and should not be over-read.** The
  script's independent pairing reproduces the span-budget filing's own depth-6 row exactly
  (4.0394 / 4.4389 / +0.3994 against that filing's line 311). Two pairings of the same npz
  files agreeing is a consistency check, not a validation of the cost model.
* **`plain-depth1`'s cost row is computed from `notul_depth1.yaml`** (`wandb.name:
  plain-depth1`); the run log's own `loop 4:6x1:4` line agrees, but the e19-era config the
  arm was launched from is gone from the tree.
* **`visible_keys_per_token` is a mean over 96 packed rows.** The distribution is not
  reported and the tail is what an OOM cares about.
* **KV bytes assume bf16 and no quantisation.** `morph/model/kv_quant.py` exists and is
  not modelled.
* **The table ranks arms at 5,000 steps.** `short-horizon-CE-is-not-a-verdict` applies:
  this is a price list, not a ranking of architectures.
