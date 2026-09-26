# Planned: does the strict slot model's gap to the plain model close from 5k to 10k

Status: planned

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
