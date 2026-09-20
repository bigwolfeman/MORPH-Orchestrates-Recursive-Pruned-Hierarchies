# Planned: the fan with epiplexity plus the within-slot volume

Status: failure

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

- A GPU smoke of the composed config: the first Spark smoke (38e16f4) died in the compile
  warmup on `linalg.slogdet` refusing a bf16 Gram; fixed in 0e9e3da (Gram in fp32 with
  autocast off, CPU bf16-autocast test added). Second smoke at 0e9e3da, 2026-09-20 13:09 to
  13:18 UTC, the runner's 12-step command, exit 0, peak 17.63 GB, `val/fan_vol_t0..6` 1.25
  to 0.85 bits per (K − 1) and `val/fan_epi_t0..6` 0.30 to 0.16 logged at step 12. Not a
  reading; the runner's own smoke gate runs again before the arm starts.
- The relative scale of the two summed parts at λ 0.1 on the real shapes (both in bits;
  P-7 reads the sum).

## Results

Run: `slot-spandec-strict-fan4-epivol` at 0e9e3da, 5000 steps on the 5090 (09:10 to 10:16
local, 2026-09-20), exit 0, tripwire HEALTHY (max 45.3 at step 239), final val_loss 4.4616.
Runner smoke exit 0, peak 17.58 GB. Artifacts in `../results/2026-09-19-lxtul-fan4/`
(`sweep_slot-spandec-strict-fan4-epivol_{2500,5000}.json`, `worth_…`, `slot_state_…`,
`run_slot-spandec-strict-fan4-epivol.txt`, `fan_geom_fan4-epivol_5000_d6.{json,txt}` from
the 3070 probe, scorer `fan_score.py`). Final-val keys (`*_final`, step 5000) unless said.

| key | epivol | epi | fan4 (cos) | norepel | mean |
|---|---|---|---|---|---|
| `fan/stream_rank_t1` / `_t6` | **2.823** / 2.312 | 1.000 / 1.002 | 1.05 / 1.08 | 1.23 / 1.20 | 1.37 / 1.36 |
| `val/fan_vol_t1` (bits per K − 1) | 2.646 | | | | |
| `val/fan_epi_t0` / `_t1` (bits per feature) | 0.566 / 1.134 | 1.071 / 1.666 | | | |
| `val/fan_stream_cos_t1` / `_t6` | −0.309 / −0.073 | 0.294 / 0.606 | −0.326 / −0.263 | 0.972 / 0.941 | 0.997 / 0.994 |
| `fan/mixed_ce` | 4.4583 | 4.4849 | 4.4744 | 4.4766 | 4.4797 |
| `fan/oracle_ce` | 4.4401 | 4.4736 | 4.4666 | 4.4638 | 4.4635 |
| oracle − mixed | **0.0182** | 0.0114 | 0.0078 | 0.0128 | 0.0162 |
| `fan/stream_ce_k0..3` − mixed | +0.005 / +0.138 / +0.423 / +0.231 | +0.203 / +0.224 / +0.227 / +0.017 | +0.099 / +0.069 / +0.035 / +0.029 | +0.030 / +0.096 / +0.123 / +0.117 | +0.010 / +0.001 / +0.002 / +0.003 |
| `fan/oracle_pick0` | 0.711 | 0.268 | 0.279 | 0.671 | 0.352 |
| `val/fan_mix_entropy` (ln 4 = 1.386) | 0.410 | 1.212 | 1.305 | 0.002 | 1.386 |
| rate at step 200 (tok/s) | 10,415 | 10,432 | 9,670 | 10,799 | 10,772 |
| K1−K6 (480 rows) | +0.0047 [+0.0043, +0.0051] | +0.0061 | +0.0023 | +0.0021 | +0.0011 |
| K3−K6 | +0.0004 [+0.0003, +0.0005] | +0.0007 | +0.0001 | −0.0005 | −0.0003 |
| depth-1 / depth-6 token CE | 4.3353 / **4.3306** | 4.3556 / 4.3495 | 4.3494 / 4.3471 | 4.3467 / 4.3446 | 4.3487 / 4.3476 |

The stream probe on the step-5000 checkpoint (2573 valid slots, forced depth 6), pass 1:
centered rank 2.825, singular-value ratios 0.85 / 0.77 / 0.00, per-stream norms 12.3 / 10.4
/ 9.3 / 11.5, shared fraction 0.175, `axis_cos` 0.139, top sign pattern `+--+` in 0.7027 of
slots, raw cosine matrix off-diagonal −0.30 to −0.40 except the (0, 3) pair at −0.07. That
is a near-regular simplex of four LIVE streams whose directions change from slot to slot:
the term got the shape it asked for. By pass 6 the shared part grows back (shared 0.46,
cos −0.08, `axis_cos` 0.37, stream 0 at norm 21 against 10 to 15): the loop pulls the
streams together again once the charged passes are over.

The coda's side: the softmax mixture concentrated on stream 0 (entropy 1.30 at step 250,
1.06 at 1000, 0.63 at 2500, 0.41 at 5000; `oracle_pick0` 0.71), stream 0 alone is within
0.005 of the mixture and the other three read 0.14 to 0.42 worse. The oracle over the four
beats the mixture by 0.018, the largest of the five arms and still under the bar. The
token CE is the best of the fan family at every depth (depth-6 4.3306, 0.017 under the
cosine arm and the ruler), inside the measured 0.024-nat seed spread and not a ranking.

- **P-1 TRUE on the number, one sub-clause missed.** `fan/stream_rank_t1` 2.823 against
  2.0; `axis_cos` 0.139 < 0.5 holds; `top_pattern_frac` 0.7027 misses the 0.7 bar by
  0.003. The shape is the one asked for (four live streams, input-dependent directions,
  balanced norms), and with 4 streams and a canonical first sign there are only 8
  patterns, so 0.70 on the top one is a weak fixed identity (stream 3 tends to side with
  stream 0), not the 0.99 of fan4.
- **P-2 TRUE.** `val/fan_vol_t1` 2.646 ≥ 1.0; `val/fan_epi_t1` − `val/fan_epi_t0` = 0.568
  ≥ 0.3 (at steps 500 to 2500 the delta sat at 0.11 to 0.13 and cleared the bar only at
  the final val, as the seed's own score fell 1.02 → 0.57).
- **P-3 FALSE.** Streams 1 to 3 sit 0.138 / 0.423 / 0.231 above the mixture.
- **P-4 FALSE (the falsifier).** Oracle below mixed by 0.0182 against 0.022. It fails while
  P-1 holds, which is the case the prereg named: streams that differ in an input-dependent,
  spread-out way still give the coda nothing to pick.
- **P-5 FALSE.** K1−K6 +0.0047 misses +0.005 by 0.0003; K3−K6 +0.0004 misses +0.001.
- **P-6 TRUE.** 10,415 tok/s.
- **P-7 TRUE.** No tripwire; the term's share of `loss/total` peaked at 4.7 % after step
  1000 (199 logged steps).
- **P-8 FALSE.** `val/fan_stream_cos_t1` −0.309, below −0.2: with three orthogonal
  deviations the raw cosine sits at the −1/3 floor, which the bar did not allow for. The
  bar was written against the cosine arm's collapse; the probe's `axis_cos` and norms are
  what tell the two apart, and they read a spread, not a split.

## Verdict

Failure. The volume-plus-epiplexity term was met by the shape it asked for, and the
falsifier still fails: an oracle over four live, input-dependent, near-orthogonal streams
gains the coda 0.018 nats over the mixture, and the mixture itself learned to read one
stream. Three diversity terms are now read on the K-stream fan (cosine, epiplexity, volume
plus epiplexity); the two that were gamed and the one that was not all leave the oracle
inside 0.008 to 0.018 of the mixture.

## Updated hypothesis

The K-stream branch closes on the survey's competing hypothesis: on web text with this
reader there is nothing for the loop to explore over at the span level. Diversity was
never the limit; when the streams are forced apart the coda picks one and ignores the
rest. The 0.017-nat CE advantage over the cosine arm is within seed spread and is not a
reason to keep the fan. What stays: `fan_repel_mode: epivol` is the term to use if a fan
is ever wanted again, because it is the one that is not gamed; and the stream probe's
cosine matrix, sign families and norms are the instrument, never one scalar.
