# Planned: the fan with an epiplexity diversity term

Status: planned

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

- A GPU smoke of the composed config: attempted on the Spark from the synced working tree
  before the queue line is added; the runner's smoke gate is the second check.
- The gradient magnitude of the epi term against the cosine's at λ 0.1 on the real
  shapes; P-7 reads it.
- Whether `fan_epi_t{t}` on bf16 autocast carries enough precision: the score runs in
  double on the readout, the deviations are cast up from the carrier's dtype.

## Results

(to be filled after the run; predictions above are frozen)
