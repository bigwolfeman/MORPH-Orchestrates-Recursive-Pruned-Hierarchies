# Planned: the fan with epiplexity plus the within-slot volume

Status: planned

Date: 2026-09-20 (frozen before any GPU step of the arm; a 12-step smoke of the config is
the only run that may precede it). Arc: the LXTUL fan (`2026-09-19-lxtul-fan4.md`,
`2026-09-20-lxtul-fan4-epi.md`).

| arm | config | one factor against |
|---|---|---|
| `slot-spandec-strict-fan4-epivol` | `tul_slot_spandec_strict_fan4_epivol.yaml` | `slot-spandec-strict-fan4-epi` — `tul.fan_repel_mode` epi → epivol |

## Question

The epi term was paid off by one input-dependent line per slot (`fan4-epi` at step 1000:
epi score 1.67 bits per feature, within-slot rank 1.001). When the term also pays for the
K − 1 within-slot directions, do the streams spread, and does the coda then have
something to choose between?

## Hypothesis

Same as the epi prereg's, one rung up: the cosine and the epi terms each had a floor a
degenerate shape reaches (a constant axis; one input-dependent line), and the sum of the
epi score with the within-slot volume has none that is content-free, because fixed axes
score 0 on epi and a line scores near 0 on volume. The competing hypothesis is unchanged:
the streams will spread and the oracle gap over the mixture will stay inside 0.02 nats.

## Predictions (frozen)

At the FINAL val (`*_final` keys at step 5000; the periodic val at 4750 is the fallback)
on `slot-spandec-strict-fan4-epivol`, scored by
`lab/experiments/results/2026-09-19-lxtul-fan4/fan_score.py` and the stream probe.

- **P-1 (the shape).** `fan/stream_rank_t1` > **2.0** of 4 (fan4 1.05, fan4-epi 1.00 at
  step 1000). **65 %.** The stream probe on the step-5000 checkpoint (48 rows, depth 6)
  reads `axis_cos` < **0.5** and `top_pattern_frac` < **0.7** at pass 1.
- **P-2 (both parts act).** At 5000 `val/fan_vol_t1` ≥ **1.0** bit per (K − 1) and
  `val/fan_epi_t1` exceeds `val/fan_epi_t0` by ≥ **0.3** bits per feature. **60 %.**
- **P-3 (no constant offset).** Every `fan/stream_ce_k{i}` within **0.03** nats of
  `fan/mixed_ce`. **50 %.**
- **P-4 (THE FALSIFIER).** `fan/oracle_ce` below `fan/mixed_ce` by more than **0.022**
  nats (fan4 0.008, norepel 0.013, mean 0.016). **30 %.** If it fails while P-1 holds, the
  K-stream branch closes with three diversity terms read: streams that differ in an
  input-dependent, spread-out way still give the coda nothing to pick.
- **P-5 (depth).** Token K1−K6 > **+0.005**; K3−K6 > **+0.001**. **25 %.**
- **P-6 (cost).** **8,500 to 11,000** tok/s at step 200 (epi 10,432). **80 %.**
- **P-7 (scale and stability).** No tripwire; `fan/repel_weighted` below **10 %** of
  `loss/total` in magnitude at every logged step after 1000 (epi: 0.6 % at step 250,
  1.6 % at step 1060). **70 %.**
- **P-8 (the cosine).** `val/fan_stream_cos_t1` between **−0.2 and 0.9**. **60 %.**

**NOT predicted:** the arm's absolute CE, other K, other λ or η, transfer to math or Sudoku.

## Method

5000 steps on the 5090 queue, inserted directly after `slot-spandec-strict-fan4-epi` (the
fan block stays together), the runner's 12-step smoke first, readouts as for fan4. The
stream probe runs on the 3070 on the step-5000 checkpoint. Scored with `fan_score.py`.

## Not verified before launch

- A GPU smoke of the composed config on the Spark, recorded here before the queue line.
- The relative scale of the two summed parts at λ 0.1 on the real shapes (both in bits;
  P-7 reads the sum).

## Results

(to be filled after the run; predictions above are frozen)
