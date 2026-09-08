# E16 probe logs

The four `training.grad_probe_path` JSONL files (15-20 MB each, ~6,400 rows x 200 keys) live in
`ignored/experiment-artifacts/2026-09-08-arc-e16/probe_<arm>.jsonl` (gitignored) and, as the source,
`/home/wolfe/morph-scratch/arc/results/2026-09-08-arc-e16/`. `score_e16.py` reads them from there.
