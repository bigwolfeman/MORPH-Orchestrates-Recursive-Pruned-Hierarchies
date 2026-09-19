# Sample-oracle gate results

Record: `../../failures/2026-09-18-sample-oracle-gate.md`. Instrument
`lab/divergence/sample_oracle_probe.py` (99bd7c9), run 2026-09-19 on the Spark against
`slot-spandec-strict_step_5000.pt`, 96 rows, N = 16, sigma {0.3, 1.0}, entry depths {1, 6},
exit depth 6, seed 0.

- `sample_oracle_strict_5000.json`: per-cell det / mean / oracle(N) / gain(N) with row
  bootstraps, `cos_exit` for the entry cells, and the arm's geometry keys.
- `sample_oracle_strict_5000.txt`: the run log with the printed table.
- Per-(row, span, sample) CE sums: `morph-scratch/arc/results/sample_oracle_strict_5000.units.npz` (private).
