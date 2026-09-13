# TUL as memory: token CE against inference cost

Every slot arm in the arc has been scored on loop CONTRIBUTION — the K-curve, the worth
profile, the per-pass cotangents — and the answer has come back the same twelve times:
the passes are worth 0.0005 to 0.0033 nats. That is a verdict on the LOOP. It is not a
verdict on the SLOT, and the two have been read as one number for a month.

The slot is also a MEMORY: a fixed-size cell per span that a later token reads instead of
re-reading the span. A memory is not priced in nats. It is priced in nats per unit of
inference cost, and nobody had written that table. This page defines the cost model and
holds the table.

The table is produced by `lab/divergence/memory_cost_table.py` and by nothing else. Every
number below comes from a file named beside it. No number here was typed from memory.

## The cost model

Three numbers per GENERATED token at decode. All three are arithmetic from the arm's own
composed Hydra config, plus ONE measurement on real packed rows (the mean span length and
the mean visible key count). `P`, `C`, `D` are `n_prelude`, `n_core`, `n_coda`; `T` is the
eval loop depth; `K` is `prefix_k`; `span_len` is the measured mean tokens per span.

### 1. `block_passes_per_token`

Transformer block applications charged to one generated token. The token always pays
`P + D`. What it pays for the CORE and for the slot CELLS is what the forwards differ by.

| forward | block passes per generated token |
|---|---|
| plain (`mode: plain`) | `P + C·T + D` |
| slot loop (`mode: slot_loop`) | `P + D + (K·(P + D) + C·T) / span_len` |
| paid loop / `loop_reads_tokens` | `(P + C·T + D)·(1 + K/span_len)` |

The slot-loop row is the whole argument in one line. A slot's cells pay the prelude and
the coda once each (`K·(P + D)`), and the core runs `T` times over ONE compact state per
slot (`C·T`) — and that cost is amortised over every token of the span the slot serves.
The token itself never enters the core. So the slot loop is the only row whose cost per
token FALLS as spans get longer.

The span decoder (`tul.spandec`) is charged 0. It is a training-only scorer: it is not in
the deployed forward at all (`morph/model/tul_spandec.py`, `_ternary_exclude` on every
leaf). That is stated here rather than silently assumed.

### 2. `visible_keys_per_token`

Mean attention key positions one decode step may read, MEASURED on real packed rows
through the SHIPPED allow-relation builders (`tg_strict_allow`, `tg_allow_mask`,
`span_allow_mask`) — never a re-derivation. A second copy of an allow relation is how two
paths drift apart, and this one would drift silently into a cost advantage.

The relation read is the CODA's, because a generation step recomputes the whole stack for
the new position and the widest stage decides what that position can see. Under `strict`
the prelude is same-span-only and the coda adds the earlier slot cells, so the coda
relation is the union; under `tg_restrict` both stages carry the same relation; a plain
model has one relation everywhere.

### 3. `kv_bytes_per_token`

`kv_bytes_per_position · visible_keys_per_token`, with

    kv_bytes_per_position = 2 · (P + D) · n_kv_heads · d_head · 2 bytes      (bf16)

The CORE contributes nothing persistent. MORPH recomputes K/V from the current carrier at
every iteration — that is the same invariant that stops the slot loop from using the token
path's active-set shrinking (`lab/runtime-invariants.md` §6b) — so there is no core KV to
cache.

## The table

Run: `lab/divergence/memory_cost_table.py`, 2026-09-13, `--rows 96 --batch 3`, on the
validation stream. CE is the per-token mean over the **491,520 stream positions shared by
all seven arms**, paired on `tok_index`, so arms that cut the same stream into different
rows still pair at the TOKEN level. Artifacts:

* `lab/experiments/results/2026-09-13-tul-as-memory/memory_cost_table_5000.json` / `.md`
* `..._5000_core_mismatch.json` / `.md` — the same run with the recovered-fraction column
  printed under `--allow-core-mismatch` (read the caveat below before using that column)

Every arm is at 5000 steps, seq 1024, `ternary_scale_mode: norm_match`, `4:6:_:4` blocks.
CE is scored at each arm's TRAINED depth: 1 for the two depth-1 plain arms, 6 for the rest.

| arm | config | CE @depth | passes/token | keys/token | KV B/token | budget recovered |
|---|---|---|---|---|---|---|
| `plain-depth1` | `notul_depth1` | 3.9612 @1 | 14.00 | 512.5 | 4,198,400 | 1.196 † |
| `budget-web-full` | `budget_web_full` | 4.0394 @6 | 44.00 | 512.5 | 4,198,400 | 1.000 (defines) |
| `plain-coda-matched` | `plain_coda_matched` | 4.0933 @1 | 14.00 | 512.5 | 4,198,400 | 0.865 † |
| `slot-spandec-strict-coretok` | `tul_slot_spandec_strict_coretok` | 4.2783 @6 | 10.59 | 64.4 | 527,359 | 0.402 † |
| `slot-spandec-strict` | `tul_slot_spandec_strict` | 4.3466 @6 | 10.59 | 64.4 | 527,359 | 0.231 † |
| `slot-spandec-mask` | `tul_slot_spandec_mask` | 4.3470 @6 | 10.59 | 64.4 | 527,359 | 0.230 † |
| `budget-web-span` | `budget_web_span` | 4.4389 @6 | 44.00 | 12.9 | 105,343 | 0.000 (defines) |

Cross-check, unplanned and worth keeping: this script's independent pairing reproduces
the span-budget filing's own depth-6 row exactly — 4.0394 / 4.4389 / +0.3994 against
`lab/experiments/failures/2026-09-11-arc-span-budget.md` line 311. Two different pairings
of the same npz files agree to four decimals.

Measured inputs: span length 20.08 tokens on the slot arms' packed rows, 19.74 on
`budget-web-span`'s (same rule, different packing); `kv_bytes_per_position` 8,192 on every
arm (`n_kv_heads` and `d_head` are shared).

## What the table says

**The slot arms are the cheapest thing in the table, by a lot.** 10.59 block-passes per
generated token against the plain depth-1 control's 14.00 and the depth-6 plain's 44.00,
and 64.4 visible keys against 512.5 — an 8.0x cut in attention context and a 8.0x cut in
KV bytes. That is the memory working: a token reads 64 positions, of which its own span is
about 20 and the rest are the slot cells that stand in for every earlier span.

**And they are 0.39 nats behind the plain depth-1 control at 0.76x its cost.** `plain-
depth1` reads 512 keys and gets 3.9612; `slot-spandec-strict` reads 64 and gets 4.3466.
Nothing here says the slot is free. It says the slot is a compression with a price, and
this is the first time that price has been written next to what it buys.

**`budget-web-span` is the reading that changes the argument.** It is the same corpus with
NOTHING crossing a span boundary, and it is 0.092 nats WORSE than `slot-spandec-strict`
while spending 4.2x the block-passes. So the two prefix cells per span do carry real
cross-span information — the K-curve saying the PASSES are worth 0.002 nats was never a
statement about the CELLS. Read that against the measured cross-span budget of 0.40 nats
(`lab/experiments/failures/2026-09-11-arc-span-budget.md`).

**`coretok` is the best slot arm in the table** (4.2783, 0.068 ahead of `strict`) at
IDENTICAL inference cost: `tul.core_token_aux` is a training-only second pass. Its loop
still reads 0.0005 nats. A training-time change that moves the memory's quality at zero
decode cost is exactly what this table is for.

## The caveat that decides how the last column is read

`budget-web-full` and `budget-web-span` (2026-09-11) run `model.core_impl: parcae` — a
dense softmax core, never ternarised — while every slot arm runs the MORPH ternary core.
Their ABSOLUTE CE is therefore not comparable to a slot arm's, and the denominator they
define (`CE_spanlocal − CE_full` = 4.4389 − 4.0394 = **0.3994**) is a scale measured on a
different model. That is why `plain-depth1` scores 1.196 — a fraction above 1.0 is the
mismatch showing itself, not an arm beating the ceiling.

† marks every cell that inherits it. The script REFUSES to print the column by default and
prints the config difference it found; the numbers above come from the
`--allow-core-mismatch` artifact.

The fix is queued, not argued: `plain-span-local` — the slot arms' OWN core with
`model.span_mask: span` — in `lab/experiments/planned/2026-09-13-arc-tul-as-memory.md`.
With that arm the denominator is measured on the same model and the column stands on its
own.

## What this page is not

Not a wall-clock measurement: `block_passes_per_token` counts block applications, and a
block application on 64 compact cells is not the same wall-clock as one on 1152 positions.
Not a K-curve. Not a claim that any arm is good. It is a table, and the ranking it
produces is the thing to argue about.
