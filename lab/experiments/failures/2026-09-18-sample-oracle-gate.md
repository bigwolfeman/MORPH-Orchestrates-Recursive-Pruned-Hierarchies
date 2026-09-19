# Sample-oracle gate: does sampling around the slot loop find alternatives the point state loses?

Status: failure (P1 and P3 fell; the branch-opening conditions were not met)

Date: 2026-09-18. Instrument: `lab/divergence/sample_oracle_probe.py`. Hosts: Spark (real
checkpoint), no training. Arm: `slot-spandec-strict` @ 5000 (the strict ruler, coda reach
`all`). Written before the probe ran on any checkpoint.

## Question

The latent-exploration literature (survey:
`docs/references/looping-depth/latent-exploration/2026-09-18-latent-exploration-survey.md`)
proposes that a loop earns depth by holding several candidate continuations and narrowing
them, and that a single deterministic trajectory collapses to one (PLR Theorem 4.4,
`D(T) = L^(2T) D(0)`; GRAM's deterministic-guidance ablation at 0.00). Parallel Test-Time
Scaling (You et al., ACL 2026) measures this without training: draw N latent trajectories
by perturbing the state, score the best of N after seeing the answer (coverage@N, the
oracle-over-samples), and compare to the one deterministic trajectory. This gate asks that
question of the strict slot loop as it is, before any K-stream arm is built.

Concretely: if N = 16 draws of noise on the slot ENTRY, run through the loop at the trained
depth, produce exit states whose best-per-span coda CE beats the deterministic exit by a
useful margin, then there are alternatives worth exploring and a selector is the problem.
If the draws collapse to one exit (the contraction the theorem predicts) or their oracle
gain is no better than noise on the EXIT alone (a reader effect, not a loop effect), the
exploration branch closes on this corpus at this scale.

## Hypothesis

The strict slot loop is a contraction around a point estimate (per-pass displacement
saturates after pass 1; loop-diagnostics 2026-09-14 read the strict loop's attractor score
at 0.82, no spectral gap). Entry noise is contracted away, so N samples at the exit are
nearly one sample, and the oracle over them is close to the deterministic CE. Whatever
oracle gain appears comes from the READER seeing a spread of exits (noise on the exit alone
gives as much or more), not from the loop turning entry variation into alternatives.

## Predictions (frozen)

Notation: `det` = deterministic coda CE (token-weighted, tokens of spans 1..end attributed
to the slot that precedes them); `oracle(N)` = per-span minimum over the first N samples of
the span's summed CE, token-weighted; `gain(N) = det - oracle(N)`; `sigma` = noise scale
relative to each slot's own RMS; `cos_exit` = mean pairwise cosine of the exit states
across the 16 samples, over valid slots.

- **P1 (collapse).** Entry noise, sigma 0.3, depth 6: `cos_exit >= 0.98`. At sigma 1.0:
  `cos_exit >= 0.90`.
- **P2 (small entry gain).** Entry noise, depth 6: `gain(16) < 0.02` nats at sigma 0.3 and
  `< 0.05` at sigma 1.0, while the mean-over-samples CE at sigma 1.0 sits `> 0.10` nats
  ABOVE `det` (noise hurts on average and the oracle recovers little).
- **P3 (reader, not loop).** At depth 6 and matched sigma, exit-noise `gain(16)` is at least
  2x the entry-noise `gain(16)` (both sigmas).
- **P4 (no growth with depth).** Entry-noise `gain(16)` at depth 6 minus at depth 1 is
  `<= 0.01` nats at both sigmas. Passes 2-6 do not turn entry variation into alternatives.
- **P5 (sanity).** `oracle(1)` equals the mean-over-samples CE to 1e-6 (it is one sample);
  `gain(N)` is non-decreasing in N; mean-over-samples CE `>= det` at every sigma.

What would OPEN the branch instead: entry-noise `gain(16) >= 0.05` nats at depth 6 with
sigma <= 0.3, `cos_exit < 0.90` (the samples are genuinely different exits), and the gain
growing from depth 1 to depth 6 by `> 0.02` (P4 fails). All three together; any one alone
is a reader or noise effect.

## Method

- Checkpoint `slot-spandec-strict/step_5000.pt` (Spark copy, md5 checked against the
  5090's). Config `tul_slot_spandec_strict`, `model.use_kernels=false`. Eval rows: the OWT
  validation stream the sweeps use, seq 1024, `--rows 96 --batch 4` (24 batches). Smoke
  first at 24 rows.
- Depth forced through `tul.slot_mean_depth` exactly as `core_depth_sweep.py` and
  `hop_distance_probe.py` do; depth 0 refused.
- Entry noise: `model.core_init.forward` is wrapped so its output `e0` (the slot loop's
  entry state, `[B, S, ..., C]`) becomes `e0 + sigma * rms_slot(e0) * eps`, `eps ~ N(0, I)`
  drawn per sample per slot, applied at VALID slots only (pad slots stay at their zero
  entry). This is the same hook `basin_map.py` uses.
- Exit noise: `model.tul.prefix_project` is wrapped so `h_slots` becomes
  `h_slots + sigma * rms_slot(h_slots) * eps` before projection. Same scale rule, same
  valid-slot mask. Depth 6 only (the exit state is the loop's output; the noise does not
  pass through the loop).
- Per sample the whole forward is `core_depth_sweep.ce_maps` → `model.tul_forward_ablated`
  so every TG mask is the model's own.
- Grid: `det` at depths {1, 6}; entry noise at sigma {0.3, 1.0} x depth {1, 6} x N = 16;
  exit noise at sigma {0.3, 1.0} x depth 6 x N = 16. Seed 0; the noise RNG is a CPU
  generator seeded per (variant, sigma, depth, sample) so every forward is reproducible.
- Aggregation: span index per token from `hop_distance_probe.span_index_from_layout`; the
  tokens of span j (j >= 1) are attributed to slot j-1. Span 0 (no preceding slot) is
  excluded. Per (row, span): summed CE per sample; `oracle(N)` = min over the first N
  samples; row-level sums feed `_stats.paired_bootstrap_ci` against `det` (95 %, 2000
  resamples). `cos_exit` is read from the `h_slots` the exit-noise wrapper receives, in
  the entry-noise runs (no noise added there; spy only).
- Output: `lab/experiments/results/2026-09-18-sample-oracle-gate/sample_oracle_strict_5000.json`
  and a printed table; per-(row, span, sample) CE sums to an `.npz` beside it.
- Verdict rule: P1-P5 scored individually; the branch stays closed unless the three
  "OPEN" conditions all hold.

## Risks

- The relative-RMS noise scale may be badly calibrated for the HC carrier (n = 4
  streams): sigma 0.3 might be negligible or catastrophic. The mean-over-samples CE at each
  sigma is reported so a mis-scaled sigma is visible; the grid is not extended after the
  fact.
- A frozen coda reads a perturbed exit as "confidently wrong" (the-reader-was-the-limit,
  2026-09-18: -4.6 nats for a cell the coda was not trained on). Exit-noise oracle gains
  are therefore a lower bound on what an adapted reader would recover; the gate compares
  entry to exit under the SAME reader, which cancels this.
- 96 rows is 1/5 of the sweep's 480; CIs are wider. The gate is sized to detect a 0.05-nat
  gain, not a 0.005-nat one.

## Results

Run 2026-09-19 07:40-08:35 UTC on the Spark, `slot-spandec-strict@5000`, 96 rows, N = 16,
seed 0. Artifacts: `../results/2026-09-18-sample-oracle-gate/sample_oracle_strict_5000.{json,txt}`;
the per-(row, span, sample) sums in `morph-scratch/arc/results/sample_oracle_strict_5000.units.npz`
(1.8 MB, private).

| cell | det | mean of 16 | gain(1) | gain(4) | gain(16) [95 %] | cos_exit |
|---|---|---|---|---|---|---|
| entry, sigma 0.3, depth 1 | 4.2661 | 4.2668 | −0.0007 | +0.0066 | **+0.0117** [+0.0115, +0.0120] | 0.954 |
| entry, sigma 0.3, depth 6 | 4.2646 | 4.2651 | −0.0003 | +0.0063 | **+0.0109** [+0.0106, +0.0112] | 0.962 |
| entry, sigma 1.0, depth 1 | 4.2661 | 4.2738 | −0.0080 | +0.0116 | **+0.0251** [+0.0241, +0.0261] | 0.670 |
| entry, sigma 1.0, depth 6 | 4.2646 | 4.2703 | −0.0056 | +0.0122 | **+0.0246** [+0.0236, +0.0257] | 0.720 |
| exit, sigma 0.3, depth 6 | 4.2646 | 4.2654 | −0.0007 | +0.0080 | **+0.0144** [+0.0140, +0.0148] | — |
| exit, sigma 1.0, depth 6 | 4.2646 | 4.2752 | −0.0100 | +0.0133 | **+0.0296** [+0.0282, +0.0311] | — |

Scorecard:

- **P1 FAILS.** `cos_exit` 0.962 at sigma 0.3 (predicted ≥ 0.98) and 0.720 at sigma 1.0
  (predicted ≥ 0.90). Six passes leave the samples genuinely different; the loop does not
  contract entry noise to one exit. The PLR Theorem 4.4 collapse does not describe this loop.
- **P2 half.** Gains small as predicted (0.0109 < 0.02; 0.0246 < 0.05). The mean-over-samples
  penalty at sigma 1.0 is +0.0057, not the predicted > 0.10: noise on the cell barely hurts.
- **P3 FAILS as written.** Exit-noise gain exceeds entry-noise gain at both sigmas (1.32x
  and 1.20x), in the predicted direction but short of the 2x bar.
- **P4 HOLDS.** gain(16) at depth 6 minus depth 1: −0.0008 (sigma 0.3), −0.0005 (sigma 1.0).
  Passes 2-6 turn none of the entry variation into alternatives.
- **P5.** The first clause was mis-specified in the prereg (`oracle(1)` is sample 0, not the
  mean over samples; they differ by 0.0002-0.0044 here, as they must). The sanity it meant
  holds: gain is non-decreasing in N in every cell and mean ≥ det in every cell.
- **Branch-opening conditions:** none met. gain(16) at sigma ≤ 0.3 is 0.011 (bar 0.05);
  cos_exit at sigma 0.3 is 0.96 (bar < 0.90); growth with depth is negative (bar > 0.02).

## Verdict

Filed under `failures/` because P1 and P3 fell, and read as a closed gate: sampling around
the strict slot loop finds no alternatives worth a selector on this corpus at this scale.
The mechanism the prereg guessed (collapse) is wrong. The mechanism the data show is
reader-side: at sigma 1.0 the exits differ by cosine 0.72 after six passes and the coda's
CE moves by +0.006 on average and −0.030 at best-of-16. **The cell's direction is close to
irrelevant to this reader.** That is the same fact as the-reader-was-the-limit (2026-09-18:
−4.6 nats frozen, +0.177 adapted) from the sampling side, and it is why exit noise
recovers as much as entry noise: the loop adds nothing to the diversity, and the reader
would not use the diversity if it did.

## Updated hypothesis

1. Sampling and selection (Parallel-TTS, GRAM, PLR-style streams) cannot pay on the strict
   slot loop while the coda is the reader: the reader is near-insensitive to the cell's
   direction, so a better sample is not a better read. Any LXTUL arm must be scored with an
   ADAPTED reader (`tul-code-target-uf` shape) or its oracle is bounded by this 0.03.
2. The loop is not a strong contraction here (cos 0.72 survives six passes at sigma 1.0).
   The register's rank-1.24 collapse was therefore not the loop squeezing distinct seeds
   together; the seeds were never made distinct in a direction the reader reads.
3. Next: repeat the two entry cells on `tul-code-target-uf@30000` (adapted reader) before
   any fan arm trains. If gain(16) there exceeds 0.05 with cos_exit < 0.90, the fan has a
   ceiling to reach for; if not, the LXTUL queue lines come out.
