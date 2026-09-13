# Planned: the register's READER, and a target that is not the next span

Status: planned

Date: 2026-09-13 (frozen before any GPU step of any of the three arms; no smoke of any of
them exists at filing time). Arc:
[`2026-09-04-loop-contribution-arc.md`](2026-09-04-loop-contribution-arc.md). Design note:
[`2026-09-13-register-reader-and-downstream-target.md`](../../../.agents/notes/proposed/architecture/2026-09-13-register-reader-and-downstream-target.md).

Three arms, two independent questions, one build.

| arm | config | one factor against |
| --- | --- | --- |
| `slot-register-m4-reader` | `tul_slot_register_m4_reader.yaml` | `slot-register-m4` — `tul.spandec_reads_cells` false → true |
| `slot-spandec-strict-off2` | `tul_slot_spandec_strict_off2.yaml` | `slot-spandec-strict` — `tul.spandec_target_offset` 1 → 2 |
| `slot-spandec-strict-off3` | `tul_slot_spandec_strict_off3.yaml` | `slot-spandec-strict` — `tul.spandec_target_offset` 1 → 3 |

Recipe, shared with every arm of the arc: 5,000 steps, seq 1024, batch 6, seed 1, ramp
1,000, `norm_match`, `core_fixed_point_lambda` 1.0, prune/carve/route off, retention off,
`tul.tg_geometry: strict`, `tg_coda_prefix_reach: all`, `loop_reach: 0`, `mux_beta: 0`,
`spandec` on at J 32. The register arms add `slot_cells: 4` and `prefix_k: 4`.

## The two defects

**The register grades the weakest reader of its own cells.** `slot-register-m4` gives a
span four mutable looped cells and then hands every reader between the loop and the write
`_reg_cells.mean(dim=2)` — one vector, one target per span, the object the one-cell ruler
already had. The register note says so and files the cross-attending decoder as its own
follow-up; its Risks section names the mean as the arm's most likely reason to read a flat
K-curve. The prereg for the register is explicit about when this arm matters: if its P-1
holds and P-3 fails, "the lane left is the target ... and the mean-graded decoder is the
first thing to remove".

**Nothing has ever asked z for a span that is not the next one.** The 2026-08-14 JEPA
screen said the informative target sits two to three spans downstream. Every span-decoder
arm has graded z on span s+1, which is also the span the coda predicts best unaided: the
cross-span budget ([`failures/2026-09-11-arc-span-budget.md`](../failures/2026-09-11-arc-span-budget.md),
491,520 paired tokens at 5,000 steps) is a **0.958**-nat spike at a span's FIRST position
on top of a **flat 0.315** at every offset eight or more tokens in.

**Horizon is not offset, and that is the whole point of the pair.**
`slot-spandec-strict-h3` set `tul.spandec_horizon: 3` and decoded spans s+1, s+2 AND s+3 as
one concatenated causal run at 3x the decoded positions — the next span was still in the
target. It read K1−K6 +0.0016, K3−K6 −0.0000, `all_slots` 0.1887, CE −0.0093 [−0.0116,
−0.0073] against strict, and its gain was CE at offsets 3+ with the far worth bins
UNCHANGED (P-f FALSE,
[`failures/2026-09-12-arc-strict-geometry.md`](../failures/2026-09-12-arc-strict-geometry.md)).
Offset MOVES the target instead of widening it: at k = 2 or 3 with H = 1 the next span
leaves the objective entirely, at the ruler's cost. That arm has never run.

## Questions

1. Does a decoder that READS the four cells separately, rather than their mean, make the
   cells worth having — and does it make any pass after the first worth having?
2. Does a target two or three spans downstream ask z for the flat, far part of the
   cross-span budget that a next-span target cannot ask for?

## Hypotheses

**Arm 1.** The register's capacity is real and its objective is not. Four cells graded by
one mean is one target per span, so the gradient that reaches the cells is no richer than
the ruler's; a decoder that attends to the cells separately gives each cell its own error
signal at every decoded token. If the mean is the bottleneck, `val/slot_cell_eff_rank`
rises and `spandec_ce` falls.

**Arm 2 and 3.** The next-span target is dominated by the 0.958-nat first-position spike,
which the coda's own token path largely covers. A target three boundaries out can only be
served by content that survives a boundary, which is the part of the budget nothing has
asked for.

**What the theory says, and it is not a prediction of success.** The information view
([`2026-09-13-information-view-of-the-slot-loop.md`](../../../.agents/notes/proposed/architecture/2026-09-13-information-view-of-the-slot-loop.md))
says a DETERMINISTIC loop cannot add information, so loop value comes only from the
reader's observation — a RELAY (content moved into a window the reader could not see) or
EXTRACTABILITY (content re-arranged into a form the reader's bounded computation can use).
All three arms keep `tg_coda_prefix_reach: all`, so the coda already sees every cell and
`mi_relay_redundant` applies: **no relay is available to any of them.** The span decoder is
an auxiliary scorer, not the coda, so a better decoder changes the OBJECTIVE's shape and
not the reader's window. The theory's prediction is therefore P-5 below: K3−K6 stays under
0.002 on all three arms with high probability. If any of them clears 0.01 with a paired CI
clear of zero, the first move is to run the strict leak gate
(`tests/test_tul_strict_geometry.py`), not to believe the number.

## Method

Same runner, same scorers, same 480 validation rows as every arm in the arc.

### Readout

1. `core_depth_sweep.py --depths 1,2,3,6,9,12,16 --rows 480` → token K1−K6, K3−K6,
   `spandec_ce`, paired bootstrap over rows. The partner `slot-spandec-strict` reads token
   K1−K6 **+0.0016 [+0.0013, +0.0019]**, K3−K6 **+0.0002**.
2. `worth_profile.py --rows 192 --modes auto` → zero / shuffle / wrong_seed / all_slots by
   offset bin. On a strict arm `all_slots` equals `zero` by construction. The partner's
   `all_slots` total is **0.1865**. The offset-0 bin is the open defect the register panel
   carries: on math the cells HURT offset 0 by about **2 nats**.
3. `val/slot_eff_rank`, `val/slot_pairwise_cos` (all 64·M cells) and
   `val/slot_cell_eff_rank`, `val/slot_cell_pairwise_cos` (WITHIN a slot). The CPU-fixture
   init floor is 1.2383 / 0.8443.
4. Depth-6 CE paired on 480 rows against the arm's one-factor partner.
5. `tul/spandec_ce` with `tul/spandec_target_offset` beside it. **The offset arms'
   `spandec_ce` is a CE over span s+k and is comparable with NO offset-1 arm's column.**
   The offset now travels with the column in the trainer's val stats and in
   `core_depth_sweep.py`, so a scorer cannot read the two as the same number.
6. Wall clock and tok/s at step 200. The partner `slot-spandec-strict`: 3,337 s,
   **11,759 tok/s**; the queue's rate floor is 8,086.
7. The matched-compute row, on every slot filing: `plain-depth1` at 14 block-passes per
   token is 0.25 nats AHEAD of `slot-spandec-mask` and 0.33 ahead of the mask ruler at 5k.

### What the builds are

**`spandec_reads_cells`.** Each `_SpanDecBlock` gains a cross-attention between its causal
self-attention and its MLP (`morph/model/tul_spandec.py::_SpanDecCross`). Queries are the
decoder's own token states; keys and values are the slot's M cells, through the same
`_readout` that makes `z`. The memory is the object `TULSlots.prefix_project` writes into
the coda — the think-once stack's output where `cond_layers > 0`, because `_tul_cond_apply`
runs before the cells are formed — so the decoder and the coda read one thing. The z
conditioning stays the MEAN. The cross output projection is ZERO-init and the cross weights
come from a third private generator with the global RNG snapshotted and restored around
their construction, so at step 0 the arm's loss, `spandec_ce` and every shared gradient are
BIT-IDENTICAL to `slot-register-m4`'s. Refused at `slot_cells == 1` and with `spandec:
false`.

**`spandec_target_offset`.** Slot s's decoder decodes span s+k and only span s+k, through
the existing `span_slots(shift=k)` primitive (which `spandec_horizon` already used for its
blocks s+1..s+H). The last k−1 slots of a row have no span s+k and are masked to −100 as
pad slots are. Teacher forcing, cost and every other knob are unchanged. Offset and horizon
compose (spans s+k .. s+k+H−1) and no arm runs the combination. Refused below 1, at or past
the derived slot budget (config time AND at the layout), with `spandec: false`, and with
`spandec_per_pass` / `oracle_z` / `coda_span_heads`, each of which defines "the next span"
for itself and would silently disagree with the decoder.

### The gate, built in this change

* `tests/test_tul_spandec_reads_cells.py` — **35 passed**.
* `tests/test_tul_spandec_target_offset.py` — **38 passed**.
* Whole suite at the commit: **1,654 passed, 9 skipped, 1 xfailed** (exit 0).
* OFF bit-identity is MEASURED, not argued: `lab/divergence/spandec_off_pin.py` ran on the
  parent commit `f89256d` (this change's four source files stashed) and on this tree, four
  fixtures at M = 1 and M = 4, and loss / logit sum / `spandec_ce` / grad sum / key count
  agree to the last printed digit. The four rows are pinned as literals in the test file.
* Fifteen source-level sabotages, each anchored to exactly one occurrence, applied to the
  shipped source: **15/15 CAUGHT**, after three MISSED on the first pass and exposed three
  real gaps in the tests (a one-layer decoder fixture that could not see a private-stream
  leak; a uniform-span fixture that silently overwrote a mis-targeted slot; and no test at
  all reaching `tul_setup.py`'s wiring). A sixteenth patch was withdrawn: it turned out to
  be an INERT rewrite, measured, not a sabotage.

## Predictions (frozen)

Written before any GPU step of any arm. Probabilities are the builder's.

- **P-1 (the reader moves the within-slot rank).** `slot-register-m4-reader`'s
  `val/slot_cell_eff_rank` at 5,000 is **higher than `slot-register-m4`'s by more than
  0.3** (of a maximum of 4). **55 %.** Reasoning: a per-cell error signal at every decoded
  token is the first force in this tree that rewards the cells for DIFFERING; the mean
  rewards only their average. Against it: the shared loop and the shared write are the
  collapse mechanism and neither changes here. Residual: 35 % within ±0.3, 10 % lower.
- **P-2 (the reader buys the decoder something).** `tul/spandec_ce` at 5,000 is **at least
  0.05 nats better** than `slot-register-m4`'s, token-count-weighted. **70 %.** Reasoning:
  it is strictly more capacity in the reader with the same target, and reader capacity is
  the cheapest thing to buy. This prediction is nearly free and it is here as the
  NON-VACUITY check: if it fails, the cross-attention is not being used at all and P-3 and
  P-4 say nothing.
- **P-3 (the passes finally read something).** `slot-register-m4-reader`'s token K1−K6 is
  **above 0.005**, against the ruler's +0.0016. **20 %.** Reasoning: this is the second of
  the two factors the register note named, and the first (capacity) is being run beside it.
  Against it: the information view says a better AUXILIARY reader cannot change the coda's
  observation, and the coda is what the K-curve measures. Residual: 60 % in [0.000, 0.005],
  20 % negative.
- **P-4 (the per-pass curve).** `slot-register-m4-reader`'s K3−K6 is **above 0.002**.
  **12 %.** Reasoning: no arm in the arc has cleared it, and the one that did (prev+reach1,
  +0.0042) did it by cutting the CODA's window, which this arm does not touch.
- **P-5 (the theory's prediction, and it covers all three arms).** All three arms read
  **K3−K6 below 0.002**. **80 %.** Reasoning: `mi_relay_redundant` — the coda reads every
  cell directly on all three, so no relay exists, and the measured per-pass reader headroom
  is 0.001 to 0.003 nats. This is the prediction that the information view is on the hook
  for. If it fails, run the strict leak gate first.
- **P-6 (the offset arms cost CE at 5k).** Both `-off2` and `-off3` are **WORSE than
  `slot-spandec-strict` on depth-6 token CE, token-paired on 480 rows**, by 0.00 to 0.06
  nats. **60 %.** Reasoning: the next span's first tokens lose their direct z target, and
  that is where the 0.958-nat spike lives. This is the price of the question, not a
  failure. Residual: 25 % better than strict, 15 % worse by more than 0.06.
- **P-7 (the offset arms move the FAR worth bins).** `slot-spandec-strict-off2`'s
  `worth_profile` `all_slots` at offset bin 16+ is **at least 0.02 nats above**
  `slot-spandec-strict`'s. **30 %.** Reasoning: this is the prediction `-h3` failed (P-f
  FALSE) with the next span still in its target; removing the near span is the change that
  could make it true. Against it: `-h3` failing suggests the profile's far bins are set by
  the coda's geometry and not by the decoder's target at all. **This is the arm's reason to
  exist and it is the one to score first.**
- **P-8 (offset 3 is not simply better than offset 2).** `slot-spandec-strict-off3` does
  NOT beat `slot-spandec-strict-off2` on depth-6 token CE by more than 0.02 nats. **70 %.**
  Reasoning: the JEPA screen's "two to three" is a range with no measurement inside it, and
  three boundaries out the conditional p(span s+3 | z, its prefix) may be close to the
  unconditional one. If this fails, distance is a live dial and the sweep continues.
- **P-9 (the far target does not go inert).** `slot-spandec-strict-off3`'s `tul/spandec_ce`
  is **at least 0.10 nats better than the same decoder reading a zeroed z**. **65 %.** This
  is the arm's own non-vacuity check: if the decoder learns to ignore z, the term trains
  nothing and `-off3` is a slower ruler with a noisier objective. Measured by one extra
  eval pass with `plan_mode: zero`.
- **P-10 (the offset arms cost no rate).** Both offset arms clear **11,000 tok/s** at step
  200 against the partner's 11,759. **85 %.** Reasoning: identical shape, identical
  decoded positions, one fewer supervised slot per row. They should be a hair FASTER.
- **P-11 (the reader's rate).** `slot-register-m4-reader` clears **8,500 tok/s** at step
  200. **60 %.** Reasoning: two cross-attentions of J=32 queries onto M=4 keys per slot is
  about 3 % of a decoder block's own FLOPs, and `slot-register-m4` has itself never run —
  so this prediction inherits the register's unmeasured rate. Arithmetic, not a
  measurement.

## Binding

If **P-2 holds and P-3 fails** — the reader is used, the decoder gets better, and the coda's
K-curve does not move — that is a direct measurement of the information view's central
claim on this tree: improving an AUXILIARY reader cannot make the loop's passes matter,
because the passes' value is bounded by the CODA's observation. The lane then moves off the
objective entirely and onto the coda's window (the relay arms), which is the only thing
that has ever produced a forced K-curve (+0.0163 at prev+reach1).

If **P-1 and P-3 both hold** — the cells separate AND the passes read something — the
register's flat result was the mean, the capacity reading survives, and the next arm is
per-cell targets (cell i decodes token bucket i), which the register note already names.

If **P-1 fails** — the cells collapse under a per-cell reader too — the collapse is the loop
and the shared write, and no target change will fix it. That closes the register lane.

If **P-7 holds** — the far worth bins move — the arc has, for the first time, an objective
that reaches the flat 0.315-nat part of the cross-span budget, and the next question is how
far out it keeps paying (the offset sweep continues past 3).

If **P-7 fails and P-6 holds** — the offset arms cost CE and move no worth bin — the
next-span target was not the reason the far budget is unreached, and the remaining
explanation is the channel, not the objective: what z can carry, not what it is asked for.

If **P-9 fails** — `-off3`'s decoder ignores z — do not read `-off3`'s other numbers at all.
An inert auxiliary term is a noise source, and `-off2` becomes the only offset reading.

## Not verified before launch

* **No GPU step of any of the three arms.** No smoke, no wall clock, no memory figure.
  Every cost number is arithmetic.
* **`slot-register-m4` itself has never run.** `slot-register-m4-reader` is one factor
  against an arm with no measurements, so until the partner runs, the reader arm has no
  baseline. It must be queued and read as a PAIR.
* **Neither `spandec_reads_cells` nor a non-unit `spandec_target_offset` has trained a
  single step.** The forward and backward were exercised at the real `d_model` 1024 on a
  256-token CPU batch for all three configs (composed through Hydra +
  `build_tul_runtime`), and every test is at `d_model` 64.
* **The cross-attention's cost is not in `_tul_layer_passes`.** That metric counts BLOCK
  passes and the number of blocks is unchanged, so the reader arm's price is slightly
  under-reported by the flop proxy. Named, not fudged.
* **The offset arms' `spandec_ce` has no baseline in the ledger.** It grades a different
  span from every earlier column. The only like-for-like comparison is `-off2` against
  `-off3`.
* **P-9's zeroed-z control is not built.** It is one extra eval pass with the existing
  `plan_mode: zero`, and it has not been run or scripted.
* **The 2.0-nat offset-0 defect on math is a register-panel finding, not a web-text one.**
  It is listed in the readout because the register arms carry it, and no web-text
  measurement of that bin exists for a register model.
* **`worth_profile.py` has never run on a `spandec_reads_cells` model.** It measures the
  coda, which this change does not touch, so it should be unaffected — should, not
  verified.
* **Seed 1, 5,000 steps, seq 1024, one scale**, like every other arm in the arc. Short
  horizon CE does not rank arms and this prereg does not use it to.

## Results

(to be filled after the runs; predictions above are frozen)
