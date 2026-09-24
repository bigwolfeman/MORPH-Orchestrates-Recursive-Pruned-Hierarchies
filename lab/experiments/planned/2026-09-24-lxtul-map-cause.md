# Planned: what causes the slot map's 0.87 (map-cause interventions I-0, I-1, I-2)

Status: planned

Date: 2026-09-24 18:00 (frozen before any GPU step of I-0, I-1 or I-2).
Parent: [`2026-09-24-lxtul-stage3-map.md`](2026-09-24-lxtul-stage3-map.md) (Stage 3; its
arm C `lxtul-e4probe-map` is the reference here).
Account under test: [`../../theory/tul_exploration/READ-BEFORE-TUNING-THE-LOOP-MAP-0.87-IS-THE-INJECTION-FLOOR.md`](../../theory/tul_exploration/READ-BEFORE-TUNING-THE-LOOP-MAP-0.87-IS-THE-INJECTION-FLOOR.md).
Wolfe's go: 2026-09-24 ("We should test I-1 and I-2. If you really think we need I-0 do it
too."; earlier the same day: "the map behaviors are an effect, not a cause").

## Question

The slot map reads 0.870–0.883. `DiagonalInjection` alone sets a floor of
`sqrt((704 + sum A^2)/1024)` = 0.864–0.867, with `A` at its 0.447 init in every model
read. The plain loop that earns depth sits 0.05–0.10 above the same floor. The Lean
account says: the slot trajectory settles from its entry because a source is re-supplied
every pass and the coda needs nothing kept across passes; a settled trajectory's gradient
cannot train the blocks' response (theorem 4); so the map reads the floor, and 0.87 is an
architecture default. Three tests, each on a different link:

- **I-0 (no training).** Where content MUST go through the loop, do the blocks respond?
- **I-1.** Move the floor and nothing else. Does the map follow the floor?
- **I-2.** Take the per-pass source away after pass 0. Does the loop then keep what it
  was given, through `A`, and does keeping it earn depth?

## Hypothesis

The account, as its author states it. I-0: on `slot-spandec-strict-prev-reach1` (the
loop is the only relay; content h spans back arrives at pass h−1) the map reads above the
floor, and more so for slots with more spans behind them. I-1: the map follows the floor
down to about 0.79. I-2: `A` rises and the map rises with it, uniformly, and the coda
still gets no depth value, because keeping the entry is memory, not computation.

## Arms and readings

| id | what | config | reference |
|---|---|---|---|
| I-0 | `core_map_fd.py` fp32, 96 rows, depth 6, no training | `tul_slot_spandec_strict_prev_reach1` @5000 (`slot-spandec-strict-prev-reach1`) | its own floor (read from its `A`); ruler `slot-spandec-strict` +0.013 |
| I-1 `lxtul-e4probe-inj9` | `model.injection_channels: all`, `injection_all_decay: 0.9`: floor at init 0.787 | `tul_slot_spandec_strict_e4probe_inj9.yaml` | `lxtul-e4probe-map` (Stage 3 arm C) |
| I-2 `lxtul-e4probe-once` | `tul.slot_source_once: true`: passes t >= 1 keep the decay, drop `dt*e`, the x0/bigram terms and the LXTUL-E code | `tul_slot_spandec_strict_e4probe_once.yaml` | `lxtul-e4probe-map` |

I-1 and I-2 compose `e4probe_map` (row hinge 0.98, tail 100 @ 1.1), so a map that rises
is not held down by e4probe's 0.9 hinge. 5000 steps from scratch, seed 1, one trainer at a
time.

Readings: `injection.log_A` (context-channel mean, and the other channels on I-1) and the
floor it implies; `core_map_fd.py` fp32 at depth 6, mean over passes 1–5 of
`operator.rms_vjp`, per-position quantiles and `frac_gt_1`; "the blocks' part" = map minus
floor; the Stage 2 scorer with `--ref` = arm C (coda mix CE @6, K1−K6, width gain);
sweeps; notul pairing; the sustained tripwire. Not built, so not read: the per-pass
cotangent gain along the loss direction that the account also names.

## Predictions

I-0:

- **P0-1.** prev-reach1's map, passes 1–5, minus its floor >= +0.03: **45 %.**
- **P0-2.** The per-slot gain rises with the slot's index in the row (Spearman rho between
  slot index and the per-slot gain, pooled over passes 1–5, >= 0.3): **35 %.**

I-1 (inj9 against arm C):

- **P1-1.** Its map within 0.02 of its OWN floor (from its trained `A`): **65 %.**
- **P1-2.** Its map <= 0.82: **60 %.**
- **P1-3.** Coda K1−K6 <= 0.003: **85 %.**
- **P1-4.** Coda mix @6 within ±0.01 of arm C: **60 %.**

I-2 (once against arm C):

- **P2-1.** Mean context-channel `A` >= 0.7 at step 5000: **45 %.**
- **P2-2.** Map >= 0.93: **40 %.**
- **P2-3.** Coda K1−K6 >= 0.005, CI clear of zero (the account says no): **25 %.**
- **P2-4.** Coda mix @6 minus arm C <= +0.02: **50 %.**

Stability, the sustained tripwire HEALTHY: I-1 **85 %**, I-2 **70 %.**

## Verdict rules

The account is SUPPORTED if P1-1 holds and P2-1 holds, and at least one of P0-1 / P0-2
holds. It is REFUTED on a link if: I-1's map returns to 0.85–0.90 (the loss chooses about
0.88: P1-1 and P1-2 both fail); or I-2's `A` stays below 0.5 while the map stays at 0.87;
or I-0 reads within +0.015 of the floor with no slot-index trend (the map does not reflect
work even where work is forced).

The positive this line exists for is P2-3: depth value on the deployed coda from a loop that
has something to keep. If P2-3 holds, the experiment succeeds whatever the account's
verdict.

No clause passes on a cosine.

## Method

`tul.slot_source_once` is built in `morph/model/transformer.py` (`DiagonalInjection.decay`,
`_apply_core_step(source_decay_only=...)`, `_tul_core`) and `morph/model/tul.py`, tested in
`tests/test_tul_source_once.py` (12 tests; eight sabotages caught). I-0 runs as soon as
the GPU lock is free between Stage 3 steps. I-1 and I-2 run after the Stage 3 chain and
after the full test suite passes: 30-step smokes, then inj9, then once, then
`core_map_fd` on both beside arm C, the Stage 2 scorer, sweeps, notul pairing. Artifacts:
JSON and `.runlog.txt` files in `../results/2026-09-24-lxtul-map-cause/`.
