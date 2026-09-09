# Density panel results (block prune; a side result)

Record: `../../failures/2026-09-09-arc-density-panel.md`. Runner `morph-scratch/arc/run_density.sh`
at `137893f`. Arms on the Parcae-entry recipe: `density-half` and `density-quarter` (MORTAR
block prune only, from step 1,500 at interval 25 and rate 0.03; no carve, no routing). This
panel was run on a misreading of "density" (Wolfe meant the weight data type); it stays as a
block-prune side result on loop contribution.

- `sweep_<arm>_<ck>.json`: `lab/divergence/core_depth_sweep.py`, 480 rows, depths 0,1,2,3,6,9,12,16;
  per-token files and probes under `ignored/experiment-artifacts/2026-09-09-density-panel/`.
- `anatomy_<arm>.json`, `init_probe_<arm>.json`: the anatomy and init probes at 5,000.
- `score_density.py` / `score_density.txt`: the trainer's own `[prune]` density check, the E17 rule,
  the depth table, P-den-a–g; references from `../2026-09-09-arc-e19/`.
- `run_<arm>.log`, `queue_density.log`.
