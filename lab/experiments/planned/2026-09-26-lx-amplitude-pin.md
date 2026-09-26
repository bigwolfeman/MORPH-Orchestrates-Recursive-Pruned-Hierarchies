# Planned: fp01's coda K-curve with the non-injected channels pinned at their entry RMS

Status: planned

Date: 2026-09-26 16:37 (frozen before any GPU step of this probe; only the tiny-model CPU tests ran).
Sibling of [`2026-09-26-lx-amplitude-matched-k-curve.md`](2026-09-26-lx-amplitude-matched-k-curve.md),
which makes a shallow exit's code LOUDER. This one holds the size fixed from below.

## Question

Outside `DiagonalInjection`'s ctx slice (512:832 on fp01) the carry is the identity, so the LX
code re-added at every pass piles up there with depth. If every slot's non-injected channels
are rescaled to the RMS they entered the loop with, after every pass and identically at every
depth, how much of fp01's coda K1-K6 (and K3-K6) is left?

## Predictions

Mine (the builder agent's), before any number of this probe. Known from the Stage 2 log
(`/home/wolfe/morph-scratch/lxtul-10k/score.log`, point estimates): free coda CE at d3 / d6
is 4.3410 / 4.3408 at 5k and 4.0695 / 4.0692 at 10k, so the free K3-K6 is about +0.0002 /
+0.0003 and nearly all of K1-K6 (+0.0125 / +0.0177) sits between depth 1 and 2.

- **P-1.** Pinned K1-K6 <= 0.006 at 5k and <= 0.008 at 10k (at least half of the free
  value disappears).
- **P-2.** The pin costs more at depth 6 than at depth 1: pin effect @6 - pin effect @1
  >= +0.01 at both checkpoints (the coda was trained on grown outer channels).
- **P-3.** Pinned K3-K6 has a CI that includes 0 at both checkpoints.

## Reading rule

If pinned K1-K6 keeps >= 70 % of the free value, the passes' work on direction carries the
curve and amplitude is not the explanation (the amplitude-matched probe's O-1 then should
fail too). If it falls below 30 %, fp01's depth reading is mostly the code's size. Between:
report both and read it beside the amplitude-matched survive ratio.

## Method

- Instrument: [`../../divergence/lx_amp_pin.py`](../../divergence/lx_amp_pin.py); the pin is
  an eval hook at `MORPHTransformer._slot_pass_hook` (None by default; bit-identical forward
  and gradients to HEAD 36a9823 on the tiny K = 4 and K = 1 models, checked on CPU), placed
  after the LX code term of every pass. Per valid slot, per Hyper-Connection stream, the
  outer channels' RMS is set to its value at loop entry; the ctx slice and pad slots are
  untouched. Tests `tests/test_lx_concept.py`.
- Rows, statistics, self-check: as the Stage 1 scorer (480 rows, forced depths 1-6, coda CE
  under the per-span Bayes read, paired block bootstrap over 1,024-token stream blocks,
  2,000 resamples, seed 0). Checkpoints fp01 step 5000 and 10000. Chain
  `/home/wolfe/morph-scratch/concept/concept_chain.sh` (last two steps), ~7 min each.
