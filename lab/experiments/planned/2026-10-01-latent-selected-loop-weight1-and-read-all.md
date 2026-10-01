# Planned: the router-followed latent-selected loop with a lighter latent loss, or a full read

Status: planned

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
