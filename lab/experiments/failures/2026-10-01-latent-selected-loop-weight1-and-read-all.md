# Planned: the router-followed latent-selected loop with a lighter latent loss, or a full read

Status: failure

Date: 2026-10-01 16:15 CDT (frozen before either run trained).
Configs: `tul_slot_spandec_strict_fan4_all_fp01_lsel_det_rf_lam1` (latent weight 1) and
`tul_slot_spandec_strict_fan4_all_fp01_lsel_det_rf_all` (the coda reads all four final
candidates). Each is `..._lsel_det_rf` plus one key.
Parent prereg: [`2026-10-01-latent-selected-loop-router-followed.md`](2026-10-01-latent-selected-loop-router-followed.md).
Wolfe (2026-10-01): "build and smoke those changes. we will run them later tonight".

## Question

The router-followed detached loop earns depth (K1-K6 +0.0095, 2.3x the ungraded fan) and
its selection and search have value on the ledger (random pick +0.011, random walk
+0.016). But its CE on the ledger's 480 rows trails the ungraded fan by 0.155 (4.489 vs
4.334). Two causes are open: (1) the latent loss at weight 10 is a tax with nothing learned
(val R^2 -0.004; the 10x factor fan paid +0.12 of gap the same way); (2) the coda reads one
cell where the ungraded fan reads four (worth 0.17 on the ledger). Which one is the cost,
and does the loop keep earning when it is removed?

## Hypothesis

Both are part of the cost. The full read is the larger part, because the ungraded fan's
own ledger prices it at 0.17. The lighter loss recovers less, because the router CE and
the per-pass reset still need the latent head to rank cells.

## References (seed 1, 5000 steps, ledger 480 rows)

| arm | tok/s | ledger CE | gap to plain | K1-K6 | zero-cell worth |
| --- | --- | --- | --- | --- | --- |
| ungraded fan | 9797 | 4.334 | +0.254 | +0.0042 | |
| router-followed detached (parent) | 9566 | 4.489 | +0.409 | +0.0095 | +0.102 |

## Predictions (mine, orchestrator, before the runs)

Latent weight 1:

- **L-1**: no detonation. 85 %.
- **L-2**: ledger CE better than the parent's 4.489 by more than 0.05. 55 %.
- **L-3**: ledger CE within 0.03 of the ungraded fan's 4.334. 30 %.
- **L-4**: K1-K6 > 0: 60 %. K1-K6 at or above the parent's +0.0095: 30 %.
- **L-5**: worth-profile zero TOTAL > 0. 85 %.
- **L-6**: val `fan_lsel_r2` > 0.1. 10 %.
- **L-7**: tok/s within 5 % of 9797. 80 %.

Full read:

- **A-1**: no detonation. 85 %.
- **A-2**: ledger CE better than the parent's 4.489 by more than 0.05. 70 %.
- **A-3**: ledger CE within 0.03 of the ungraded fan's 4.334, or better: 45 %. Better than it
  by more than 0.0137: 15 %.
- **A-4**: K1-K6 > +0.0042 (above the ungraded fan). 50 %.
- **A-5**: ledger `random_search` > 0 with its CI above 0 (the search still pays when the
  coda reads every candidate). 50 %.
- **A-6**: ledger `single_cell` > +0.05 (the coda uses the width). 75 %.
- **A-7**: tok/s within 5 % of 9797. 80 %.

Pass, latent weight 1: L-3 AND K1-K6 > 0.
Pass, full read: A-3 (within 0.03) AND A-4 AND A-5.

## Method

- Seed 1, 5000 steps each, `runner_steps.sh` at the commit of this prereg, full read first.
  Readouts as for every fan arm (depth sweep, gap to plain, worth profile); the stage-2
  score fails on fan arms (no parallel head) and is ignored. Then the exploration ledger
  (this tree) and the repetition eval (N=32, 128 tokens) on both arms.
- Smokes at this commit, 45 steps on the 5090, 2026-10-01: weight 1 exit 0, 9253 tok/s,
  14.29 GB; full read exit 0, 9177 tok/s, 14.27 GB. Ledger smoke on the full-read smoke
  checkpoint: kind lsel_all, 14 readings, exit 0.

## Results

Both runs finished 5000 steps at dba8592, seed 1, full read first. Both tripwires read
HEALTHY. Artifacts: [`../results/2026-10-01-latent-selected-loop-weight1-and-read-all/`](../results/2026-10-01-latent-selected-loop-weight1-and-read-all/).
CE is the ledger's shipped CE on the 480 sweep rows (511089 tokens); tok/s is the mean of
the logged steps after the first two.

| arm | tok/s | ledger CE | gap to plain | K1-K6 | zero-cell worth | greedy rep-4 |
| --- | --- | --- | --- | --- | --- | --- |
| ungraded fan | 9797 | 4.334 | +0.254 | +0.0042 | | 0.523 [0.450, 0.597] |
| router-followed detached (parent) | 9566 | 4.489 | +0.409 | +0.0095 | +0.102 | 0.602 [0.537, 0.672] |
| latent weight 1 | 9818 | 4.447 | +0.366 [+0.350, +0.384] | +0.0333 [+0.0318, +0.0347] | +0.102 | 0.568 [0.501, 0.633] |
| full read | 9508 | 4.502 | +0.422 [+0.406, +0.440] | +0.0170 [+0.0161, +0.0180] | +0.115 | 0.590 [0.524, 0.651] |

CE at each forced depth (sweep, 480 rows):

| arm | K1 | K2 | K3 | K4 | K5 | K6 |
| --- | --- | --- | --- | --- | --- | --- |
| ungraded fan | 4.3385 | 4.3362 | 4.3349 | 4.3344 | 4.3342 | 4.3343 |
| parent | 4.4988 | 4.4944 | 4.4898 | 4.4894 | 4.4893 | 4.4894 |
| latent weight 1 | 4.4802 | 4.4635 | 4.4473 | 4.4468 | 4.4470 | 4.4470 |
| full read | 4.5191 | 4.5057 | 4.5026 | 4.5018 | 4.5018 | 4.5020 |

Ledger deltas (reading minus shipped, nats per token, 95 % CI):

| reading | latent weight 1 | full read |
| --- | --- | --- |
| random exit pick | +0.0093 [+0.0086, +0.0099] | n/a (no exit pick) |
| random walk | +0.0113 [+0.0106, +0.0120] | +0.0056 [+0.0051, +0.0060] |
| teacher | +0.0022 [+0.0017, +0.0028] | +0.0003 [+0.0000, +0.0006] |
| no reset | +0.0034 [+0.0029, +0.0038] | +0.0041 [+0.0037, +0.0046] |
| fixed lineage | +0.0284 [+0.0271, +0.0299] | +0.0168 [+0.0157, +0.0177] |
| single cell | +0.0165 [+0.0159, +0.0171] | +0.0691 [+0.0666, +0.0719] |
| best-of-4 oracle | -0.0263 [-0.0269, -0.0257] | +0.0031 [+0.0019, +0.0044] |

Val at step 5000: latent weight 1 `fan_lsel_r2` -0.038, teacher-router agreement 0.386,
cell spread 0.144, switch rate 0.315. Full read: R^2 +0.011, agreement 0.408, spread 0.220,
switch rate 0.265.

| prediction | reading | held |
| --- | --- | --- |
| L-1 no detonation | HEALTHY | yes |
| L-2 better than parent by > 0.05 | better by 0.042 | no |
| L-3 within 0.03 of the ungraded fan | +0.113 | no |
| L-4 K1-K6 > 0 / at or above +0.0095 | +0.0333 | yes / yes |
| L-5 zero worth > 0 | +0.102 | yes |
| L-6 R^2 > 0.1 | -0.038 | no |
| L-7 tok/s within 5 % of 9797 | 9818 | yes |
| A-1 no detonation | HEALTHY | yes |
| A-2 better than parent by > 0.05 | WORSE by 0.013 | no |
| A-3 within 0.03 of the ungraded fan | +0.168 | no |
| A-4 K1-K6 > +0.0042 | +0.0170 | yes |
| A-5 random walk > 0, CI above 0 | +0.0056 [+0.0051, +0.0060] | yes |
| A-6 single cell > +0.05 | +0.069 | yes |
| A-7 tok/s within 5 % of 9797 | 9508 (-2.9 %) | yes |

## Verdict

Failure for both arms: each misses its CE bar (L-3, A-3).

The latent weight was part of the cost, and it was holding the loop back. At weight 1 the CE
improves by 0.042 and the loop contribution is 3.5x the parent's (+0.0333, 8x the ungraded
fan's). The depth curve shows real earning over the first three passes (4.480 -> 4.447), not
a weak first pass alone: its K1 is also below the parent's. But every depth still sits above
the ungraded fan's K1 (4.339), so the arm earns depth from a worse base.

The full read hurt. The coda uses the width (single cell +0.069), yet the arm is worse than
reading the winner alone. My interpretation: after the reset, the four final candidates are
children of one winner (spread 0.22), not four independent lineages, so reading all four
adds near-copies, not the ungraded fan's independent width. Its best-of-4 oracle is +0.003:
no single child beats the set.

Repetition did not improve in either arm (intervals overlap the ungraded fan's).

## Updated hypothesis

The latent loss is a tax on the coda and a brake on the loop. It is never learned (R^2 at or
below 0.01 in all six latent-selected arms) yet the selection works without it being
learned: the router's pick, the search and the lineage all have value on the ledger. So the
teacher needs a head that RANKS cells, not a loss that PULLS cells to the target. Next
candidate: train the latent head on detached cells (no latent gradient into the loop), so
the teacher still ranks and the cells pay no tax. The remaining gap at weight 1 (+0.113 to
the ungraded fan) is open; the single-cell read (the reader-trained router's cost, +0.011)
explains only a tenth of it.
