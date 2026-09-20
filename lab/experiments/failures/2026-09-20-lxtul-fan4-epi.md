# Planned: the fan with an epiplexity diversity term

Status: failure

Date: 2026-09-20 (frozen before any GPU step of the arm; a 12-step smoke of the config is
the only run that may precede it and it produces no reading). Arc: the LXTUL fan
(`2026-09-19-lxtul-fan4.md`), whose first arm met its cosine repulsion with a constant
two-against-two split on one axis.

| arm | config | one factor against |
|---|---|---|
| `slot-spandec-strict-fan4-epi` | `tul_slot_spandec_strict_fan4_epi.yaml` | `slot-spandec-strict-fan4` — `tul.fan_repel_mode` cos → epi |

Partners already run or queued at the same commit family: `slot-spandec-strict-fan4`
(cos, e3e654d, done), `slot-spandec-strict-fan4-norepel` and `-mean` (no term; running),
`slot-spandec-strict-pk4` (width control). Note:
`.agents/notes/proposed/architecture/2026-09-20-fan-epiplexity-diversity.md`.

## Question

When the diversity term cannot be paid off by a constant offset on one axis, do the K
streams come to differ in an input-dependent, spread-out way, and does the coda then
have something to choose between?

## Hypothesis

The fan4 reading was a property of the cosine term, not of K streams through a shared
contractive core: the term's floor is reachable with rank 1 and gradient descent found
that. A term that centers, pays for rank and reads through a frozen random function of
the input has no such floor. The competing hypothesis is the survey's: there is nothing
to explore over on this corpus, so the streams will spread (the term will be satisfied)
and the oracle gap over the mixture will stay inside 0.02 nats however they spread.

## Predictions (frozen)

All at the FINAL val (the trainer's `*_final` keys at step 5000; the periodic val at 4750
is the fallback if the final one is missing) on `slot-spandec-strict-fan4-epi`, scored by
`lab/experiments/results/2026-09-19-lxtul-fan4/fan_score.py` and the stream probe.

- **P-1 (the shape).** `fan/stream_rank_t1` > **2.0** of 4 (fan4 1.05; the register
  1.24). **70 %.** The stream probe (`lab/divergence/fan_stream_probe.py`, 48 rows, depth
  6, on the step-5000 checkpoint) reads `axis_cos` < **0.5** and `top_pattern_frac` < **0.7**
  at pass 1 (fan4 0.96 and 0.99): no global axis, no fixed identities.
- **P-2 (the term acts).** `val/fan_epi_t1` at 5000 exceeds `val/fan_epi_t0` (the seed's
  own score) by at least **0.5** bits per reservoir feature. **60 %.** If it fails the
  term never got hold of the streams and P-1, P-3 and P-4 are not readings of it.
- **P-3 (no constant offset).** Every `fan/stream_ce_k{i}` within **0.03** nats of
  `fan/mixed_ce` (fan4: +0.03 to +0.10). **50 %.** Reasoning: with no constant split
  there is no offset to put a single stream off the coda's distribution; against it, a
  mixture of streams that genuinely differ is itself a state no single stream equals.
- **P-4 (THE FALSIFIER).** `fan/oracle_ce` below `fan/mixed_ce` by more than **0.022**
  nats (fan4 0.008). **30 %.** This replaces the fan4 prereg's single-vs-oracle reading,
  which conflated the offset with the choice. If it fails while P-1 holds, streams that
  differ in a learnable way still give the coda nothing to pick, and the K-stream branch
  closes with both diversity terms read.
- **P-5 (depth).** Token K1−K6 > **+0.005** at 5000 (fan4 +0.0023, family ~+0.002).
  **25 %.** K3−K6 > **+0.001**.
- **P-6 (cost).** **8,500 to 11,000** tok/s at step 200 (fan4 9,670); the F × F slogdet
  per charged pass is negligible. **80 %.**
- **P-7 (scale and stability).** No tripwire, and `fan/repel_weighted` stays below
  **10 %** of `loss/total` in magnitude at every logged step after 1000. **70 %.** If it
  trips, the fallback named in the note is `fan_repel_lambda`, not the term's shape.
- **P-8 (the cosine under epi).** `val/fan_stream_cos_t1` between **−0.2 and 0.9**: not
  at the −1/3 floor, not collapsed. **60 %.**

**NOT predicted:** the absolute CE of the arm against the ruler (a 5000-step CE cannot
rank looped arms), any K other than 4, any λ other than 0.1, transfer to math or Sudoku.

## Method

5000 steps on the 5090 queue (`recon_arms.txt`, inserted after `slot-spandec-strict-pk4`
so the fan block closes together), the runner's own 12-step smoke first, readouts as for
fan4 (sweeps at 2500 and 5000 on 480 rows, worth profile, state probe). The stream probe
runs on the 3070 on the step-5000 checkpoint, shipped by scp with an md5 check. Scored
with `fan_score.py` (wandb `*_final` keys, the sweep JSON, the RATE line).

## Not verified before launch

- A GPU smoke of the composed config: DONE on the Spark 2026-09-20 09:31 to 09:35 UTC from
  the working tree at b01564e, the runner's 12-step smoke command, exit 0, the val ran and
  `fan/epi_t0..8` logged (0.29 to 0.18 bits per feature at step 12). Not a reading; the
  runner's own smoke gate runs again before the arm starts.
- The gradient magnitude of the epi term against the cosine's at λ 0.1 on the real
  shapes; P-7 reads it.
- Whether `fan_epi_t{t}` on bf16 autocast carries enough precision: the score runs in
  double on the readout, the deviations are cast up from the carrier's dtype.

## Results

Run: `slot-spandec-strict-fan4-epi` at b01564e, 5000 steps on the 5090 (07:47 to 08:54 local,
2026-09-20), exit 0, tripwire HEALTHY (max 255 at step 239), final val_loss 4.4878. Runner
smoke exit 0, peak 17.56 GB. Artifacts in `../results/2026-09-19-lxtul-fan4/`
(`sweep_slot-spandec-strict-fan4-epi_{2500,5000}.json`, `worth_…`, `slot_state_…`,
`run_slot-spandec-strict-fan4-epi.txt`, `fan_geom_fan4-epi_5000_d6.{json,txt}` from the
3070 probe, scorer `fan_score.py`). Final-val keys (`*_final`, step 5000) unless said.

| key | fan4-epi | fan4 (cos) | norepel | mean |
|---|---|---|---|---|
| `fan/stream_rank_t1` | 1.0001 | 1.05 | 1.23 | 1.37 |
| `val/fan_stream_cos_t1` / `_t6` | 0.294 / 0.606 | −0.326 / −0.263 | 0.972 / 0.941 | 0.997 / 0.994 |
| `val/fan_epi_t0` / `_t1` / `_t6` (bits per feature) | 1.071 / 1.666 / 1.404 | | | |
| `fan/mixed_ce` | 4.4849 | 4.4744 | 4.4766 | 4.4797 |
| `fan/oracle_ce` | 4.4736 | 4.4666 | 4.4638 | 4.4635 |
| oracle − mixed | **0.0114** | 0.0078 | 0.0128 | 0.0162 |
| `fan/stream_ce_k0..3` − mixed | +0.203 / +0.224 / +0.227 / +0.017 | +0.099 / +0.069 / +0.035 / +0.029 | +0.030 / +0.096 / +0.123 / +0.117 | +0.010 / +0.001 / +0.002 / +0.003 |
| `fan/oracle_pick0` | 0.268 | 0.279 | 0.671 | 0.352 |
| `val/fan_mix_entropy` | 1.212 | 1.305 | 0.002 | 1.386 |
| rate at step 200 (tok/s) | 10,432 | 9,670 | 10,799 | 10,772 |
| K1−K6 (480 rows) | **+0.0061 [+0.0057, +0.0066]** | +0.0023 | +0.0021 | (Spark) |
| K3−K6 | +0.0007 [+0.0005, +0.0009] | +0.0001 | −0.0005 | (Spark) |
| depth-1 / depth-6 token CE | 4.3556 / 4.3495 | 4.3494 / 4.3471 | 4.3467 / 4.3446 | (Spark) |

The stream probe on the step-5000 checkpoint (2573 valid slots, forced depth 6): at pass 1
the per-stream mean norms are 14.5 / 0.1 / 0.1 / 18.9, centered rank 1.000, shared fraction
0.987, `axis_cos` 0.125, top-direction sign patterns `+---` in 1347 slots and `+++-` in
1226, raw cosine between streams 0 and 3 0.148. Read together: in each slot ONE stream
carries the state (stream 0 in about half the slots, stream 3 in the rest, chosen by the
input) and the other three are near zero. The seed already leans that way (norms 14.0 /
4.8 / 4.7 / 15.7 at pass 0) and pass 1 finishes it; the dead streams regrow to 1.7 by pass
6. That is the loophole named in the note's amendment, `d_i(n) = c_i · v(n)`, realised as a
one-hot stream per slot: every stream's deviation is a learnable function of the seed
(which stream fires is), so the per-stream epiplexity is high, and the within-slot rank is
exactly 1.

- **P-1 FALSE.** `fan/stream_rank_t1` 1.0001 against the 2.0 bar. The probe's two
  sub-clauses hold on the numbers (`axis_cos` 0.125 < 0.5, `top_pattern_frac` 0.52 < 0.7),
  but for the wrong reason: there is no global axis and no fixed identity because the
  live stream is chosen per slot, not because the streams spread.
- **P-2 TRUE.** `val/fan_epi_t1` − `val/fan_epi_t0` = +0.595 bits per feature (bar 0.5).
  The term took hold and was paid.
- **P-3 FALSE.** Three of four streams sit 0.20 to 0.23 nats above the mixture (the dead
  streams, and stream 0 where stream 3 is the live one); stream 3 is within 0.017.
- **P-4 FALSE (the falsifier).** Oracle below mixed by 0.0114 against the 0.022 bar,
  between fan4's 0.008 and the no-term controls' 0.013 and 0.016.
- **P-5 FALSE on the letter, half true.** K1−K6 +0.0061 clears +0.005 (the first fan arm
  to do so); K3−K6 +0.0007 misses +0.001. The depth-6 CE is 0.002 WORSE than the ruler's
  (4.3495 vs 4.3474) and the depth-1 CE 0.006 worse, so the larger gap is a worse start,
  not a better end; the K3−K6 clause exists for exactly this and it fails.
- **P-6 TRUE.** 10,432 tok/s.
- **P-7 TRUE.** No tripwire; the term's share of `loss/total` peaked at 2.1 % after step
  1000 (199 logged steps scanned).
- **P-8 TRUE.** `val/fan_stream_cos_t1` 0.294.

## Verdict

Failure. The term acted (P-2) and was met by a shape that carries no diversity: a one-hot
stream per slot. The falsifier (P-4) and the mechanism clause (P-1) both fail, and the
oracle over streams adds 0.011 nats over the mixture, the same 0.008 to 0.016 every fan arm
reads with or without a diversity term.

## Updated hypothesis

Any diversity term that is a per-stream scalar over the batch can be paid by gating, and
any pairwise term with a floor can be paid by a fixed sign pattern; neither looks inside a
slot for K − 1 directions. The next arm, `slot-spandec-strict-fan4-epivol`
(`2026-09-20-lxtul-fan4-epivol.md`, queued directly after this arm), adds the within-slot
volume so a one-hot slot and a line both score near zero. If the streams then spread and
the oracle still sits inside 0.02 nats of the mixture, the K-stream branch closes on the
survey's competing hypothesis: nothing to explore over on this corpus with this reader.
