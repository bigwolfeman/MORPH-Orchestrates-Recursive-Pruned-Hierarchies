# Planned: does the strict slot model's gap to the plain model close from 5k to 10k

Status: failure

Date: 2026-09-26 03:09 (the plain resume started 03:07; frozen before its first eval and
before any 10k plain reading).
Parent: [`../successes/2026-09-25-lxtul-fp01-10k.md`](../successes/2026-09-25-lxtul-fp01-10k.md).
Wolfe's go: 2026-09-26 ("lets run it"), after the question of the loop's contribution as a
whole.

## Question

On coda tokens of the same 480 held-out rows, fp01 (strict slot geometry, the slot loop the
only cross-span channel) sits 0.265 nats behind the plain model (full attention, no slots) at
5k. fp01's slot channel is worth +0.265 (zero ablation) at 10k and grew 0.07 from 5k. Does the
gap to plain shrink from 5k to 10k?

## Method

Faithful resume of `plain-panel-norm-match/step_5000.pt` (`notul_panel_norm_match`, the
matched control of the 2026-09-13 panel) to 10000 at `d5ddf6e`; its wandb config and today's
composed config differ only in int/float formatting and 11 inert new keys (checked 03:05).
`core_depth_sweep.py` at 10k (480 rows, the old sweep's settings), then
`lxtul_e_notul_pair.py` pairs fp01's 5k and 10k coda tokens (`lxtul-10k/score.tokens.npz`)
with plain 5k and plain 10k.

## Predictions

- **P-1 (the gap shrinks).** fp01 10k minus plain 10k is smaller than fp01 5k minus plain 5k
  (0.265), with the 10k CI's top below 0.265: **60 %.**
- **P-2 (it shrinks a lot).** The 10k gap is <= 0.18: **25 %.**
- **P-3 (plain stays healthy).** Plain tripwire HEALTHY over 5001-10000: **80 %.**

## Verdict rules

Success if P-1 holds. A failure means the strict slot model's deficit is not closing at this
rate; the loop's depth growth does not change that reading.

## Results (filed 2026-09-26 04:11)

Plain resume 03:07 → 04:07, exit 0, 8762 tok/s (fp01: 5851), val loss 3.8395 (fp01 10k:
4.1154). Artifacts: [`../results/2026-09-26-lxtul-fp01-vs-plain-10k/`](../results/2026-09-26-lxtul-fp01-vs-plain-10k/).
Coda tokens of the same 480 rows, n = 491,520, each fp01 checkpoint paired with each plain one
(`lxtul_e_notul_pair.py`):

| | plain 5k (4.0750) | plain 10k (3.7359) |
|---|---|---|
| fp01 5k (4.3401) | **+0.2651** [+0.2612, +0.2686] | +0.6042 |
| fp01 10k (4.0685) | −0.0065 | **+0.3326** [+0.3282, +0.3369] |

- From 5k to 10k plain improved 0.339 nats and fp01 0.272. The matched-step gap WIDENED
  0.265 → 0.333 (+0.068, five times the 0.0137 seed spread measured on fp01's family).
- fp01 at 10k equals plain at 5k (−0.0065): the strict slot model is about 5k steps behind.
- Plain's own looped core earns depth too, and grows: K1−K6 +0.0528 (5k) → +0.0612 (10k),
  3.5x fp01's slot loop at 10k (+0.0177).
- Plain tripwire HEALTHY, max 29.2 at 9570.

A rough reading, from two different instruments and not scored: fp01's slot channel is worth
0.197 (5k) and 0.265 (10k) under the zero ablation (the 10k filing's addendum). Channel worth
plus the gap is 0.46 then 0.60, so the channel carries about 43-44 % of that sum at both
checkpoints while the sum grows. If that sum approximates the cross-span information the
model can use (the no-slot budget read 0.40 at 5k on another recipe), the channel keeps a
constant fraction of a growing budget.

## Clause by clause

- **P-1 FAILS.** The 10k gap is +0.3326, above 0.265.
- **P-2 FAILS.** Above 0.18.
- **P-3 HOLDS.** Tripwire HEALTHY.

## Verdict

**Failure.** The strict slot model's deficit to the plain model is not closing; it grew by
0.068 nats in 5k steps. The slot channel does work (worth 0.20-0.26 nats) and its depth use
grows, but the plain model gains more from training than the slot model does. One seed per
arm; the widening is five times the measured seed spread.

## Updated hypothesis

The cross-span information a model can use grows with training, and the 2-cell-per-span slot
channel carries a roughly constant share of it, so the absolute deficit grows. What would
change that is not more depth (the plain core already earns 3.5x more depth than the slot
loop) but a wider or better-used channel. Not yet measured: the gap at 20k, and whether the
share estimate holds under a direct budget measurement on this recipe.
