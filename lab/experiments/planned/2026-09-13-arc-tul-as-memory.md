# Planned: price the slot as a MEMORY, and measure the denominator on the right model

Status: planned

Date: 2026-09-13. The TABLE is already produced from existing artifacts and is committed
(`docs/tul-as-memory.md`,
`lab/experiments/results/2026-09-13-tul-as-memory/`); what is PLANNED here is the one arm
that removes the table's single blocking confound. Arc:
[`2026-09-04-loop-contribution-arc.md`](2026-09-04-loop-contribution-arc.md). Design note:
[`2026-09-13-tul-as-memory.md`](../../../.agents/notes/proposed/architecture/2026-09-13-tul-as-memory.md).

## What the table already says

Seven arms, 5,000 steps, seq 1024, `norm_match`, each at its TRAINED depth, CE paired on
the **491,520 stream positions all seven share** (`memory_cost_table.py`, 2026-09-13):

| arm | CE @depth | passes/token | keys/token | KV B/token |
| --- | --- | --- | --- | --- |
| `plain-depth1` | 3.9612 @1 | 14.00 | 512.5 | 4,198,400 |
| `budget-web-full` | 4.0394 @6 | 44.00 | 512.5 | 4,198,400 |
| `plain-coda-matched` | 4.0933 @1 | 14.00 | 512.5 | 4,198,400 |
| `slot-spandec-strict-coretok` | 4.2783 @6 | 10.59 | 64.4 | 527,359 |
| `slot-spandec-strict` | 4.3466 @6 | 10.59 | 64.4 | 527,359 |
| `slot-spandec-mask` | 4.3470 @6 | 10.59 | 64.4 | 527,359 |
| `budget-web-span` | 4.4389 @6 | 44.00 | 12.9 | 105,343 |

The slot arms are the cheapest row in the table — 10.59 block-passes per generated token
and 64.4 visible keys, an 8.0x cut in attention context and KV bytes against the depth-1
plain control — and they pay 0.385 nats for it. `budget-web-span`, the same corpus with
NOTHING crossing a span boundary, is 0.092 nats WORSE than `slot-spandec-strict` at 4.2x
the block-passes: **the two prefix cells per span carry real cross-span information**, and
the K-curve reading 0.002 was never a statement about the CELLS.

## The confound, and the arm that removes it

`budget-web-full` and `budget-web-span` run `model.core_impl: parcae` — a dense softmax
core, never ternarised — while every slot arm runs the MORPH ternary core. The recovered
fraction `(CE_spanlocal − CE_arm) / (CE_spanlocal − CE_full)` therefore has a denominator
(**0.3994**) measured on a DIFFERENT model. `memory_cost_table.py` refuses to print the
column unless `--allow-core-mismatch` is passed, and prints the config difference it
found; the `--allow-core-mismatch` artifact exists so the numbers are available with the
caveat attached. `plain-depth1` scoring 1.196 there is the mismatch showing itself.

## Question

What fraction of the cross-span information budget do the slot cells actually recover,
measured against a span-local control that shares the slot arms' own core?

## Method

ONE arm, one factor against the slot family's core.

| arm | config | one factor |
| --- | --- | --- |
| `plain-span-local` | `plain_span_local.yaml` (**to be written**) | `model.span_mask: span` on the slot arms' MORPH ternary core, no slots |

5,000 steps, seq 1024, batch 6, seed 1, ramp 1,000, `norm_match`, prune/carve/route off,
retention off, `mean_depth` 6. `model.span_mask: span` builds the dense per-position
compressed attention branch, the segment-reset CCA conv and value shift, and a cuttable
hash bigram — the same build `budget_web_span.yaml` uses, on a different core.

**This arm is not yet buildable and that is the first thing to check.** `model.span_mask`
currently REQUIRES `core_impl: parcae` and `use_kernels: false`
(`morph/model/CLAUDE.md`). Either that requirement is a real constraint of the span-mask
build — in which case the matched control has to come from the other direction (run the
slot family on a Parcae core, which `slot-mnext-parcae-core` already shows is flat) — or
it is a guard that can be widened with a leak test. **Resolve that before queueing, and
record which it was.** `tests/test_span_mask_leak.py` is the gate either way.

Readout: re-run `memory_cost_table.py` with `plain-span-local` as `--span-local` and the
matched full-context control as `--full`, and print the recovered-fraction column without
`--allow-core-mismatch`.

## Predictions (frozen)

- **M-1 (the slot cells recover a real fraction).** With the matched denominator,
  `slot-spandec-strict` recovers **more than 0.15** of the cross-span budget. **60 %.**
  Reasoning: the mismatched estimate is 0.231, and the mismatch inflates every morph-core
  arm (plain-depth1 reads 1.196, which is impossible). Residual: 25 % in [0.05, 0.15],
  15 % below 0.05.
- **M-2 (the register moves it).** If `slot-register-m4` runs, it recovers **more** of the
  budget than `slot-spandec-strict` does. **55 %.** Reasoning: four cells per span is four
  times the channel width at the boundary, and the channel — not the loop — is what the
  table prices.
- **M-3 (cost stays flat with M).** `slot-register-m4`'s `block_passes_per_token` is
  **below 14.00** (the depth-1 plain control's), i.e. the register is still cheaper than
  the cheapest plain arm. **70 %.** Arithmetic: `P + D + (K(P+D) + CT)/span_len` at K=4 is
  8 + (32 + 36)/20.08 = **11.39**. This is a check on the cost model, not a discovery —
  if it fails, the model is wrong.
- **M-4 (span_mask composes with the MORPH core).** `plain-span-local` builds and passes
  `tests/test_span_mask_leak.py` without a Parcae core. **40 %.** Reasoning: the guard
  exists for a reason that is not written down anywhere I could find, which usually means
  the fused kernels cannot take the relation. If this fails, M-1 is scored on a Parcae
  denominator anyway and stays caveated.

## Binding

If **M-1 holds** — the cells recover a real, matched fraction of the budget — the slot is
a working compression with a measured price, and the arc's headline stops being "the loop
is flat" and becomes "the memory works, the loop does not, and they are different
mechanisms".

If **M-1 fails** — the matched fraction is near zero — then `budget-web-span`'s 0.092-nat
deficit against the slot arms is a core artefact and the cells carry nothing. That would
be the strongest negative result in the arc and it retires the slot as a memory too.

If **M-4 fails** — the span mask cannot run on the MORPH core — the matched control has to
be built the other way (slot arms on a Parcae core) and the table's last column stays
caveated until then. Say that in the doc rather than quietly printing the number.

## Not verified before launch

* **`plain_span_local.yaml` does not exist yet.** The arm is specified, not built. It is
  queued LAST for that reason.
* **No GPU step.** The table is entirely re-read from existing sweep artifacts.
* **The cost model is arithmetic, not wall clock.** A block application on 64 compact
  cells is not the same wall clock as one on 1152 positions; `docs/tul-as-memory.md` says
  so in place.
* **`plain-depth1`'s cost row is computed from `notul_depth1.yaml`** (`wandb.name:
  plain-depth1`). The run log's own `loop 4:6x1:4` line agrees with it, but the e19-era
  config file the arm was launched from is gone from the tree.
* **`visible_keys_per_token` is measured on 96 packed rows**, not on the full validation
  set, and it is a mean — the distribution is not reported.
* **KV bytes assume bf16 and no quantisation.** `morph/model/kv_quant.py` exists and is
  not modelled.
* **Nothing here is a wall-clock or a throughput claim**, and no arm in the table was
  re-run to produce it.

## Results

(the table is filled; the matched-denominator column is not)
