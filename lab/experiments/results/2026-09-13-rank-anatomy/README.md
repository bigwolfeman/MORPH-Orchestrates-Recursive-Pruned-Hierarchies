# Rank anatomy of the TUL slot state (2026-09-13)

`lab/divergence/slot_rank_anatomy.py` at commit `f89256d`, run on the DGX Spark
(`/home/wolfe/morph-instruments/MORPH`, same sha, GB10, bf16 autocast, eager +
`tg_scoped_kernels`). Three step-5000 checkpoints, 480 identical validation rows each,
batch 3, `skip_samples=0`, forced slot depth 6 (each arm's own eval depth). 51 valid slots
per row (IQR 48..55), 24,819 slots in total, 1024 dimensions.

| arm | config | wall |
|---|---|---|
| `slot-spandec-strict` | `tul_slot_spandec_strict` | 923 s |
| `slot-spandec-strict-coretok` | `tul_slot_spandec_strict_coretok` | 951 s |
| `slot-spandec-strict-prev-reach1` | `tul_slot_spandec_strict_prev_reach1` | 1086 s |

Raw output: `<arm>.json` (every number) and `<arm>.txt` (the printed tables).
`strict-trainer-matched.{json,txt}` repeats the strict arm on the TRAINER's val recipe
(`skip_samples=50000`, batch 6, 20 batches).

## The reproduction check, and where it landed

Every batch is also scored by the model's own `tul_slot_state_probe` — the exact code the
trainer logs `val/slot_eff_rank` from. The instrument agrees with it to every printed
digit on all four runs:

| run | model's own probe | this instrument |
|---|---|---|
| strict, 480 rows | 11.9597 / 0.5635 | 11.9597 / 0.5635 |
| coretok, 480 rows | 12.1109 / 0.5121 | 12.1109 / 0.5121 |
| prev-reach1, 480 rows | 12.4135 / 0.5089 | 12.4135 / 0.5089 |
| strict, trainer-matched (skip 50000, batch 6, 20 batches) | 13.8466 / 0.5201 | 13.8466 / 0.5201 |

**It does not land on 5.7598 / 0.7104**, the figure the two 2026-09-13 pre-registrations
quote for this arm from `run_slot-spandec-strict.log`. The gap is not a stage or a readout
difference: the trainer's own probe code, on the trainer's own val stream, at the
trainer's batch size and eval count, reads 13.85 on this checkpoint. Two independent
readings in the tree corroborate the 12-18 range and none corroborates 5.76:
`slot_z_optimize`'s `z_eff_rank/loop` reads 17.66 for this arm at 5000
(`lab/experiments/results/2026-09-12-strict/`), and the same field on four sibling strict
arms reads 11.65-16.13.

**RESOLVED the same day, in [`discrepancy.md`](discrepancy.md).** The run logged from the
probe as it was BEFORE commit `7a24adf`, whose front was a bare `_tul_front` and therefore
ran a `tg_geometry: strict` model's prelude UNRESTRICTED. On the trainer's exact final-eval
rows that old code reads 5.7598 / 0.7104 and HEAD's reads 13.1183 / 0.5430, from the same
checkpoint at the same dtype; the run's val CE reproduces to 4.4249 on the same rows, so
the weights are not in it. The shipped reading is the right one, and the baseline a prereg
should use for this arm is 13.85 / 0.520 (trainer val recipe, stream offset 0).

## Stage tables

`rankRaw` is the participation ratio of the uncentered second moment, `rankCen` the same
after the mean comes out (the `val/slot_eff_rank` definition), `cosRaw` the mean cosine
between distinct slots, `pc1Cen` the top centered component's share, `||mu||/||s||` the
row mean's length against a state's, and `residPrev` the share of a state that lies
OUTSIDE the previous stage's top-8 span. Stages: `s0_seed` the slot's input value
(pre-prelude, embedding space), `s1_entry` the loop entry `core_init(e)`, `pass1..6` the
exit at each forced depth, `sX_exit` the unforced exit (bit-equal to `pass6`, asserted
every batch), `sW_write` the `prefix_project` output the coda reads (2 cells per slot).

### readout view — what `val/slot_eff_rank`, the MUX head and the span decoder see

Per-row median over 480 rows, each cell `rankRaw / rankCen / cosRaw / residPrev`.

| stage | strict | coretok | prev-reach1 |
|---|---|---|---|
| `s0_seed` | 2.38 / 4.67 / 0.56 / — | 2.19 / 5.23 / 0.61 / — | 2.35 / 4.43 / 0.57 / — |
| `s1_entry` | 2.44 / 11.05 / 0.61 / 0.98 | 3.05 / 13.05 / 0.53 / 0.97 | 3.34 / 11.17 / 0.50 / 0.98 |
| `pass1` | 2.33 / 10.22 / 0.63 / 0.44 | 2.59 / 10.76 / 0.59 / 0.51 | 3.04 / 10.88 / 0.53 / 0.43 |
| `pass2` | 2.35 / 9.92 / 0.62 / 0.34 | 2.61 / 10.18 / 0.58 / 0.37 | 2.98 / 10.57 / 0.54 / 0.39 |
| `pass3` | 2.35 / 9.77 / 0.62 / 0.33 | 2.63 / 9.92 / 0.58 / 0.36 | 2.95 / 10.28 / 0.54 / 0.38 |
| `pass4` | 2.35 / 9.68 / 0.62 / 0.33 | 2.64 / 9.80 / 0.58 / 0.35 | 2.90 / 10.08 / 0.55 / 0.37 |
| `pass5` | 2.35 / 9.61 / 0.62 / 0.33 | 2.65 / 9.75 / 0.58 / 0.35 | 2.86 / 9.95 / 0.55 / 0.36 |
| `pass6` | 2.35 / 9.58 / 0.62 / 0.33 | 2.66 / 9.72 / 0.58 / 0.35 | 2.83 / 9.86 / 0.55 / 0.36 |
| `sX_exit` | 2.35 / 9.58 / 0.62 / 0.33 | 2.66 / 9.72 / 0.58 / 0.35 | 2.83 / 9.86 / 0.55 / 0.36 |
| `sW_write` | 3.81 / 9.14 / 0.46 / 0.69 | 3.29 / 6.68 / 0.51 / 0.66 | 3.29 / 8.46 / 0.51 / 0.69 |

Pooled over all 24,819 slots (49,638 write cells), each cell `rankRaw / rankCen / pc1Cen / ||mu||/||s||`.

| stage | strict | coretok | prev-reach1 |
|---|---|---|---|
| `s0_seed` | 2.46 / 6.07 / 0.358 / 0.749 | 2.26 / 6.90 / 0.336 / 0.779 | 2.41 / 5.64 / 0.379 / 0.755 |
| `s1_entry` | 2.76 / 20.48 / 0.167 / 0.759 | 3.72 / 25.09 / 0.137 / 0.702 | 4.12 / 20.06 / 0.170 / 0.674 |
| `pass1` | 3.65 / 23.03 / 0.123 / 0.708 | 4.57 / 25.06 / 0.109 / 0.667 | 4.23 / 20.73 / 0.153 / 0.674 |
| `pass2` | 4.05 / 23.07 / 0.115 / 0.689 | 5.09 / 24.48 / 0.105 / 0.647 | 4.53 / 20.70 / 0.143 / 0.662 |
| `pass3` | 4.19 / 23.11 / 0.113 / 0.682 | 5.30 / 24.29 / 0.104 / 0.639 | 4.72 / 20.60 / 0.138 / 0.654 |
| `pass4` | 4.26 / 23.13 / 0.111 / 0.679 | 5.44 / 24.26 / 0.104 / 0.634 | 4.84 / 20.57 / 0.135 / 0.649 |
| `pass5` | 4.31 / 23.13 / 0.111 / 0.677 | 5.55 / 24.29 / 0.103 / 0.630 | 4.92 / 20.59 / 0.132 / 0.647 |
| `pass6` | 4.33 / 23.15 / 0.110 / 0.676 | 5.62 / 24.35 / 0.103 / 0.628 | 4.98 / 20.64 / 0.130 / 0.644 |
| `sX_exit` | 4.33 / 23.15 / 0.110 / 0.676 | 5.62 / 24.35 / 0.103 / 0.628 | 4.98 / 20.64 / 0.130 / 0.644 |
| `sW_write` | 7.46 / 19.42 / 0.148 / 0.562 | 5.21 / 12.57 / 0.234 / 0.631 | 5.29 / 16.70 / 0.148 / 0.627 |

### raw view — the carrier's own geometry, stream mean, no normalisation

Per-row median over 480 rows, each cell `rankRaw / rankCen / cosRaw / residPrev`.

| stage | strict | coretok | prev-reach1 |
|---|---|---|---|
| `s0_seed` | 1.50 / 2.24 / 0.58 / — | 1.48 / 2.24 / 0.60 / — | 1.50 / 2.20 / 0.56 / — |
| `s1_entry` | 1.29 / 12.75 / 0.86 / 0.99 | 1.75 / 14.95 / 0.76 / 0.99 | 1.57 / 11.45 / 0.77 / 0.99 |
| `pass1` | 1.30 / 13.66 / 0.87 / 0.26 | 1.64 / 10.35 / 0.79 / 0.39 | 1.48 / 11.81 / 0.81 / 0.28 |
| `pass2` | 1.32 / 13.41 / 0.87 / 0.23 | 1.67 / 10.21 / 0.78 / 0.30 | 1.48 / 11.85 / 0.81 / 0.26 |
| `pass3` | 1.33 / 13.32 / 0.86 / 0.23 | 1.69 / 10.17 / 0.78 / 0.30 | 1.49 / 11.73 / 0.80 / 0.26 |
| `pass4` | 1.34 / 13.27 / 0.86 / 0.23 | 1.71 / 10.13 / 0.77 / 0.30 | 1.49 / 11.62 / 0.80 / 0.26 |
| `pass5` | 1.34 / 13.20 / 0.86 / 0.23 | 1.72 / 10.05 / 0.77 / 0.30 | 1.49 / 11.47 / 0.80 / 0.26 |
| `pass6` | 1.34 / 13.17 / 0.86 / 0.23 | 1.74 / 9.99 / 0.76 / 0.30 | 1.50 / 11.37 / 0.80 / 0.26 |
| `sX_exit` | 1.34 / 13.17 / 0.86 / 0.23 | 1.74 / 9.99 / 0.76 / 0.30 | 1.50 / 11.37 / 0.80 / 0.26 |
| `sW_write` | 2.75 / 7.16 / 0.57 / 0.68 | 2.61 / 5.46 / 0.60 / 0.69 | 2.60 / 6.69 / 0.59 / 0.69 |

Pooled over all 24,819 slots (49,638 write cells), each cell `rankRaw / rankCen / pc1Cen / ||mu||/||s||`.

| stage | strict | coretok | prev-reach1 |
|---|---|---|---|
| `s0_seed` | 1.52 / 2.49 / 0.612 / 0.820 | 1.50 / 2.51 / 0.610 / 0.829 | 1.52 / 2.45 / 0.621 / 0.816 |
| `s1_entry` | 1.32 / 26.60 / 0.148 / 0.928 | 1.88 / 37.02 / 0.091 / 0.854 | 1.65 / 22.00 / 0.173 / 0.874 |
| `pass1` | 1.40 / 36.37 / 0.092 / 0.919 | 1.94 / 25.79 / 0.138 / 0.854 | 1.57 / 24.91 / 0.149 / 0.888 |
| `pass2` | 1.45 / 37.68 / 0.083 / 0.911 | 2.04 / 26.28 / 0.134 / 0.842 | 1.61 / 25.81 / 0.139 / 0.883 |
| `pass3` | 1.47 / 38.49 / 0.079 / 0.908 | 2.11 / 26.92 / 0.131 / 0.835 | 1.64 / 25.99 / 0.136 / 0.879 |
| `pass4` | 1.48 / 38.98 / 0.077 / 0.906 | 2.16 / 27.31 / 0.129 / 0.830 | 1.66 / 26.14 / 0.134 / 0.876 |
| `pass5` | 1.49 / 39.27 / 0.075 / 0.905 | 2.20 / 27.56 / 0.127 / 0.826 | 1.68 / 26.31 / 0.132 / 0.875 |
| `pass6` | 1.50 / 39.47 / 0.074 / 0.905 | 2.24 / 27.74 / 0.127 / 0.823 | 1.69 / 26.51 / 0.130 / 0.873 |
| `sX_exit` | 1.50 / 39.47 / 0.074 / 0.905 | 2.24 / 27.74 / 0.127 / 0.823 | 1.69 / 26.51 / 0.130 / 0.873 |
| `sW_write` | 3.85 / 14.73 / 0.226 / 0.700 | 3.32 / 9.57 / 0.292 / 0.732 | 3.39 / 13.69 / 0.225 / 0.724 |

## The reading

1. **It is BOTH, and the offset is the larger half.** In the carrier, a row's own mean
   carries 93 % of a state's length at the exit (per-row median `||mu||/||s||` 0.929, mean
   pairwise cosine 0.86), so the uncentered `rankRaw` reads 1.34 — near-parallel by
   construction. Taking that row mean out does not open the state up: 13.2 directions of a
   possible ~50, and 9.6 in the readout the coda actually reads. A shared offset alone
   would have left a near-full-rank residual. There is a real collapse underneath it.
2. **The prelude makes the rank; the loop does not touch it; the write halves it.** The
   seed is the narrowest stage of all (`rankCen` 2.24 per row, 2.49 pooled — `E_slot` plus
   one projected boundary token). The prelude multiplies it (2.24 → 12.75 per row, 2.49 →
   26.6 pooled). Six loop passes move it from 13.66 to 13.17. The prefix write
   then cuts it to 7.16 while DOUBLING the number of vectors (102 cells against 51 slots)
   — `W_prefix` is the narrowest thing downstream of the prelude.
3. **The loop lowers the per-row rank and raises the pooled one.** From pass 1 to pass 6
   the per-row `rankCen` falls on all three arms — raw view −0.49 strict, −0.36 coretok,
   −0.44 prev-reach1; readout view −0.64, −1.04, −1.02 — while the pool-centered figure
   over 480 rows RISES on all three (raw 36.4 → 39.5, 25.8 → 27.7, 24.9 → 26.5). The
   passes push rows apart and flatten each row. They are not idle while doing it: each pass after the first
   puts 23-36 % of its state outside the previous pass's top-8 span, and buys no rank with
   it. This is the same shape as every K-curve on this family — the loop acts, and the act
   does not become usable variety.
4. **Per-row centering is worth about 10x; the shipped readout already does part of it.**
   `lm_mixer` + `final_norm` take the pooled uncentered rank from 1.50 to 4.33 and the mean
   cosine from 0.82 to 0.46. Removing the ROW's own mean does much more: the raw per-row
   rank goes 1.34 → 13.17, the readout's 2.35 → 9.58. A reader that sees one row at a time
   and subtracts that row's mean is looking at a 13-dimensional signal where the raw
   carrier looks 1.3-dimensional. Nothing in the shipped forward does this. (The pooled
   `rankCen` column subtracts the mean of all 24,819 slots, which is a different and larger
   number — do not read it as the per-row figure.)
5. **The register (M cells per slot) is aimed at the wrong bottleneck.** A row already has
   ~50 slots and uses ~13 directions, so the state count is not the limit — there is spare
   room in the axes that exist. And `sW_write`, which already carries 102 cells per row,
   is the LOWEST-rank stage after the seed. Adding cells adds vectors into the stage that
   is losing rank, not into the one that is making it. The one arm here that moved the
   entry geometry is `coretok`: its extra token-through-core objective lifts the pooled
   entry rank to 37.0 against strict's 26.6 and drops the offset to 0.854 — and the loop
   then pulls it back down to 27.7. The lever that moves this geometry sits in the
   objective and in the prelude, not in the loop and not in the number of cells.

## What this cannot say

Eval-only, on trained checkpoints at 5,000 steps: a stage that reads low rank says these
models' states sit there, not that a model trained differently would. Nothing here is a CE
or a loop-contribution result — read it beside `core_depth_sweep.py` and `worth_profile.py`,
not instead of them. The 480-row panel uses `skip_samples=0` (the probe family's
convention), not the trainer's held-out shard; the trainer-matched run shows the two
differ by about 1.9 rank units on the strict arm. `s0_seed`'s readout column applies
`lm_mixer` + `final_norm` to a pre-prelude embedding, which is not the coda's space — its
raw row is the meaningful one. Reading 2's "the write halves it" is a raw-view,
single-pooled-set statement: [`discrepancy.md`](discrepancy.md) §2 shows most of that drop
is the offset BETWEEN a slot's two cells, and that the same 108 vectors read 12.80 once
each cell's own mean is out.
