# Planned: LXTUL-R Step 1b — the same arm at the corrected one-slot-per-pass geometry

Status: planned

Date: 2026-09-22 (frozen before any GPU step; committed before the runner line is
appended). Re-run of Step 1
([`2026-09-22-lxtul-r-step1.md`](2026-09-22-lxtul-r-step1.md), filed as a method fault:
the register's in-loop reach was one slot per pass through attention and about one slot
per LAYER through the CCA conv and the value shift, which the branch left unsegmented).
The fix (a per-slot `tg_seg` beside the relation whenever `loop_reach > 0`, reach 0
untouched) is the commit right before this file; `tests/test_lxtul_r_composition.py`
(11) asserts exact zeros beyond one slot per pass with the shipped conv.

## Question

With the leak closed, does the fan's K streams on the chain geometry (fan4-all +
`loop_reach 1` + `tg_coda_prefix_reach prev` + fixed-point 0 + renorm) earn a token
K-curve of the size the far budget allows, and do its passes recover the depth-1 cost
the geometry imposes?

## Hypothesis

Unchanged from Step 1, with one number updated by the leaky run: the geometry's depth-1
cost against fan4-all-fp0 was 0.040 nats with several slots relayed per pass; with one
slot per pass it will be larger, and the passes have strictly more to fetch. The renorm
held carried content at 90 % under the leak; the same mechanism should hold it now.

## Method

One arm, config `tul_slot_spandec_strict_fan4_all_reach1` unchanged (the fix is in the
model code, gated on `loop_reach > 0`), runner kind `slot`, sweeps at 2500 and 5000, 480
rows, OUTDIR `/home/wolfe/morph-scratch/arc/results/2026-09-22-lxtul-r-step1b`, wandb name
`slot-spandec-strict-fan4-all-reach1` (a new run id; the prior run stays as the leaky
record). Readouts and references exactly as Step 1's Method, plus:

8. The leak check on the real model: the planted probe's g = 3 row at depth 1
   (P-9 below), which the leaky arm read at 0.048.

## Predictions (frozen)

- **P-1 (A1.1, compose).** As Step 1. **95 %.**
- **P-2 (A1.2, healthy).** As Step 1. **75 %.**
- **P-3 (A1.3, the headline).** `ce_tokens K1-K6` at 5000 at or above **+0.063**.
  **25 %.** Point estimate +0.03. The leaky arm read +0.020 with less to fetch per pass;
  closing the leak raises what a pass can earn and what the geometry costs. Below +0.02:
  35 %. In [+0.02, +0.063): 40 %.
- **P-4 (A1.4, the decay).** Planted g = 2 at depth 6 keeps at least **75 %** of its
  depth-1 benefit. **55 %.** The renorm held 90 % under the leak.
- **P-5 (A1.5, CE parity).** Paired depth-6 CE vs `slot-spandec-strict` at or above
  **−0.02**. **55 %.** The leaky arm sat at −0.033; a tighter reach costs more.
- **P-6 (A1.6, candidates).** `fan/stream_rank_t1` ≥ **2.5** of 4; oracle − mixed quoted
  beside the 0.04 floor. **65 %.**
- **P-7 (A1.7, rate).** ≥ **5,677** tok/s. **70 %.** The segment reset is a mask on
  existing ops; the leaky arm ran at 7,496.
- **P-8 (A1.8, the guard).** Fraction `(0.2021 − gap_arm) / 0.1259` at or above **0.25**:
  **25 %.** At or below 0: 45 %. Depth-1 gap vs fp0 above 0.040: 70 %.
- **P-9 (the leak is closed on the real model).** Planted g = 3 at depth 1 inside
  **[−0.01, +0.01]**, and g = 3 at depth 2 above +0.02 (arrival one pass later, the
  chain's staircase). **85 %.**

## Binding

Step 1's binding, applied to this run: pass = P-1 to P-7; P-3 holds and P-4 fails → Step
2; P-3 fails with P-4 holding → the reach frame is closed for this loop and the
two-channel design is the next note; both fail → `failures/`, Wolfe decides; P-8 is
reported and qualifies a P-3 pass; P-9 fails → the run is a method fault again and
nothing else is read from it.

## Not verified before launch

- No GPU step of the corrected composition; the smoke is the first.
- The segment reset's interaction with the fan's winner replay (the coda's
  `checkpoint_blocks` path) is outside the core and untouched, but not re-measured.
- Whether the value shift is the only remaining per-layer route besides the conv: the
  tiny-model probe reads exactly 0 beyond one slot with both in place, which is the
  claim, but the probe is one row of one fixture.
