# Planned: is fp01's coda K1-K6 the loop computing, or the LX code getting louder with depth

Status: planned

Date: 2026-09-26 14:38 (frozen before any GPU step of this probe and before Step 0 ran; the unit tests
ran on the tiny CPU model only).
Protocol: `/home/wolfe/morph-scratch/tulv2/opus.md` Section C, "The single measurement to
run first" (Steps 0-2), with the calibration split of gpt.md Section C; fable.md Section C
asks for the same control.
Parent readings: [`../successes/2026-09-25-lxtul-fp01-10k.md`](../successes/2026-09-25-lxtul-fp01-10k.md)
(coda K1-K6 +0.0125 [+0.0117, +0.0132] at 5k, +0.0177 [+0.0166, +0.0188] at 10k).

## Question

LXTUL-E adds `0.1 * rms(f(h)) * u_k` at the end of every slot-loop pass. Outside the
320-channel ctx slice of `DiagonalInjection` the carry is the identity, so the code
accumulates about linearly with depth, and the stored Stage 2 arrays show the rollouts'
exit separation grow 2.56x (5k) and 2.22x (10k) from depth 1 to depth 6. The coda was
trained almost only on deep exits (P(T <= 1) is about 1.7 %). So a depth-1 exit differs
from a depth-6 exit by the passes' work AND by a quieter code, and K1-K6 cannot separate
the two. If the depth-1 exit gets a code as loud as depth 6's, how much of K1-K6 is left?

## Method

- Instrument: [`../../divergence/lx_amp_matched.py`](../../divergence/lx_amp_matched.py) (tests
  `tests/test_lx_probes.py`), reusing the Stage 1 scorer's labelled forward
  (`lxtul_e_stage1_score.forward_readings` / `score_arm`, which now also reports
  `sep_code`). Runs on the GPU chain `/home/wolfe/morph-scratch/tulv2/probe_chain.sh`,
  after the A2-10K chain ends, one job on the GPU.
- Checkpoints: `lxtul-e4probe-fp01` step 5000 and step 10000, config
  `tul_slot_spandec_strict_e4probe_fp01`.
- Multiplier a: eval only, the model's code ratio is set to `0.1 a` and restored. a = 1 is
  the trained model bit for bit (tested).
- Step 0 (CPU): the fraction of each u_k's energy, and of the code subspace's, on the
  injected ctx slice `injection.start:end` (read from the built model; 512:832 by the
  config's `channel_dims`), against the random-init basis and a 2,000-draw Gaussian null
  (expected share 320 / 1024 = 0.3125).
- Step 1: separation statistic `sep_abs`: per valid slot, the RMS over streams and
  channels of the difference between two rollouts' exit states, averaged over the 6 pairs
  and the valid slots (the Stage 1 scorer's `exit sep`). Calibration rows: the 33 packed
  validation rows right after the 480 scored rows (11 batches of 3; disjoint). For d in
  {1, 2, 3}, a_d is found by bisection on log a so that sep_abs(d, a_d) is within 0.5 % of
  sep_abs(6, 1) on the calibration rows. Then the 480 scored rows (the rows
  `core_depth_sweep.py` and the Stage 2 scorer read) are scored at (6, 1), (d, 1),
  (d, a_d) and (6, 1/a_d). Readings: K{d}-K6 = CE(d, 1) - CE(6, 1); K{d}*-K6 =
  CE(d, a_d) - CE(6, 1); survive_d = (K{d}*-K6) / (K{d}-K6) with a joint bootstrap CI.
  The code-subspace separation `sep_code` (gpt.md's code-direction component) is reported
  at every point and never fitted.
- Step 2 (reverse bracket): CE(6, 1/a_d) - CE(d, 1).
- Per-offset-bin CE (`lab/divergence/_earning.py` BINS) for every reading.
- Statistics: coda CE under the per-span Bayes read over the 4 rollouts, token-paired by
  stream index; paired block bootstrap over stream blocks of 1,024, 2,000 resamples,
  seed 0. Self-checks: the scorer's per-batch check against the forward's own
  `ce_tokens` (tol 2e-3), and the a = 1 points at depths 1 and 6 against the stored Stage
  2 arrays of the same checkpoint (`coda_idx` identical; mean CE within 5e-4, else the
  run raises).
- Expected, not graded (an instrument forecast from the stored separations): a_1 near 2.5
  at 5k and near 2.2 at 10k if separation is linear in a at depth 1.

## Predictions

From the Opus analyst file (opus.md Section C), for the 5k checkpoint:

- **O-1.** At least 60 % of fp01's K1-K6 disappears: **K1*-K6 <= 0.005** (unmatched
  0.0125). Holds if the point estimate of K1*-K6 at 5k is <= 0.005.

Opus gives no number for 10k or for d = 2, 3; they are reported under the same rules.

## Reading rules

- From opus.md Section C: if >= 70 % survives (K1*-K6 >= 0.009 at 5k), the passes do
  something beyond amplifying the code, and D4 (a context-specific code for the loop to
  integrate) moves up. If it vanishes, LX's depth is a gain knob on the code, and v2 effort
  goes to the channel (D1, D3).
- From gpt.md Section C (stated for the 10k checkpoint, K1-K6 0.0177): if dose-matched
  depth 1 recovers at least 0.012 of the 0.0177 (K1-K6 minus K1*-K6 >= 0.012), code
  accumulation is a sufficient explanation for most of the apparent depth use. If it does
  not, the result is ambiguous, because the louder depth-1 code may be outside training; it
  is not proof of iterative reasoning.
- Step 2 is read beside Step 1: if quieting depth 6's code to depth 1's separation costs
  about as much as K1-K6 (reverse_1 near +K1-K6 in size), the coda's reading depends on
  the code's size, whichever depth made it. If CE(6, 1/a_1) stays near CE(6, 1) while
  K1*-K6 is small, the loud code on a 1-pass state is doing the work and the passes add
  little.
- Step 0: a trained u_k share below the null's 2.5 % quantile means training moved the
  codes OFF the injected slice, toward the channels where they accumulate (opus.md).
- No clause passes or fails on a separation or a share alone; they are diagnostics.
