# Planned: what causes the slot map's 0.87 (map-cause interventions I-0, I-1, I-2)

Status: failure

Date: 2026-09-24 18:00 (frozen before any GPU step of I-0, I-1 or I-2).
Parent: [`2026-09-24-lxtul-stage3-map.md`](../successes/2026-09-24-lxtul-stage3-map.md) (Stage 3; its
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

## Results (filed 2026-09-25 08:52)

Artifacts: [`../results/2026-09-24-lxtul-map-cause/`](../results/2026-09-24-lxtul-map-cause/).
Chain at 966ea9b. Full suite on that commit: 2650 passed, 56 skipped, 1 xfailed. I-0 ran
2026-09-24 18:55; I-1 and I-2 ran 5000 steps each with exit 0 (6,175 and 6,239 tok/s),
tripwire HEALTHY on both (max 19.8 and 52.2). Scorer against arm C (`lxtul-e4probe-map`):
480 rows, 501,106 coda tokens, self-check max |dev| under 1e-6.

Map (fp32, eval, depth 6, `operator.rms_vjp`) and floor from each checkpoint's own `A`
(`map_vs_floor.runlog.txt`):

| model | mean A | floor | pass 0 | passes 1–5 | mean 1–5 − floor |
|---|---|---|---|---|---|
| I-0 prev-reach1 | 0.432 | 0.864 | 0.923 | 0.889 → 0.880 | +0.019 |
| arm C map (ref) | 0.437 | 0.864 | 0.871 | 0.871–0.872 | +0.007 |
| I-1 inj9 | 0.743 (ctx ~0.43, the rest ~0.88) | 0.773 | 0.783 | 0.781 | +0.008 |
| I-2 once | 0.443 | 0.865 | 0.873 | 0.871–0.872 | +0.006 |

| clause | reading, 95 % CI | verdict |
|---|---|---|
| P0-1 prev-reach1 map − floor >= +0.03 | +0.019 | failed |
| P0-2 slot-index trend rho >= 0.3 | −0.011 | failed |
| P1-1 inj9 map within 0.02 of its floor | +0.008 | held |
| P1-2 inj9 map <= 0.82 | 0.781 | held |
| P1-3 inj9 coda K1−K6 <= 0.003 | +0.0010 [+0.0008, +0.0013] | held |
| P1-4 inj9 coda @6 within ±0.01 of C | +0.0046 [+0.0024, +0.0066] | held |
| P2-1 once mean A >= 0.7 | 0.443 | failed |
| P2-2 once map >= 0.93 | 0.871 | failed |
| P2-3 once coda K1−K6 >= 0.005 | +0.0016 [+0.0014, +0.0018] | failed |
| P2-4 once − C coda @6 <= +0.02 | +0.0159 [+0.0136, +0.0181] | held |
| stability HEALTHY, I-1 / I-2 | HEALTHY / HEALTHY | held |

More readings: par K1−K6 inj9 +0.0113, once +0.0078 (C −0.0002). Exit separation @6:
inj9 0.29, once 0.18 (C 0.26). Against notul on 491,520 identical tokens: C +0.2496, inj9
+0.2543, once +0.2655.

## Verdict

Failure: the account is not supported (P2-1 failed) and it is REFUTED on the I-2 link by
its own rule. With the source gone after pass 0, `A` stayed at 0.443 and the map at 0.871;
the loop did not take up keeping the entry, and the coda paid 0.016 for the lost source.

What stands: **0.87 is an architecture default** (I-1). Moving the injection's floor to
0.773 moved the map to 0.781, the blocks' part unchanged (+0.008 against +0.007), at no
depth value and 0.0046 of coda CE. I-0 is between the account's two outcomes: +0.019 over
the floor (less than its +0.03, more than the ruler's +0.013) with no slot-index trend.

The Stage 3 filing the same night found the lever the account ranked fourth: with the
fixed-point term off the coda earns K1−K6 0.0123 while the map moves only +0.025 over the
floor ([`../successes/2026-09-24-lxtul-stage3-map.md`](../successes/2026-09-24-lxtul-stage3-map.md)).

## Updated hypothesis

The typical gain is the wrong reading of what the slot loop does. It is dominated by an
injection that does not train (`A` moves by at most 0.03 in every arm, whatever the
objective asks), and depth use appeared in the one arm whose map barely moved. What moved
in that arm is the trajectory: the rollouts separate 13x more by depth 6. The next
instrument reads motion across passes and what the coda takes from it, not the map's gain.
