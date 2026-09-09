# ARC E17 results: the Sudoku-Extreme depth grid

Record: `../../successes/2026-09-08-arc-e17-sudoku-depth-grid.md`. Runner:
`morph-scratch/arc/run_e17.sh` at `7538d78`; the plain arm's sweeps are the re-sweep at
`53b2472` (`run_e17_resweep_then_e18.sh`), after the stride bug in the sweep's plain rows.

- `sweep_sud-<arm>_<ck>.json`: `lab/divergence/olympiad_sweep.py` on all 3,000 held-out
  boards at depths 1,2,3,4,6,9,12,16, with per-board sums (`doc_ans_correct`, `doc_band`).
- `score_e17.py` / `score_e17.txt`: the grid (bucket × T), P17a–h, paired bootstraps.
- `run_sud-<arm>.log`: the trainer logs; `resweep_6000.log`: one re-sweep log.
- Probes (47 MB each): `ignored/experiment-artifacts/2026-09-08-arc-e17/probe_sud-<arm>.jsonl`.
- The shifted plain-arm JSONs from before the fix stay out of the record under
  `morph-scratch/arc/results/2026-09-08-arc-e17/stride_bug/`.
