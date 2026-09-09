# ARC E18 results: the slot-channel width sweep

Record: `../../planned/2026-09-08-arc-e18-slot-width-sweep.md` (Method amendment 1: rows
do not pair across cuts; the plain path). Runner `morph-scratch/arc/run_e18.sh` at
`cc8512b`; the k8@5000 and both notul sweeps ran the fixed sweep tool (`3a48d05`) and carry
per-token CE files; the others are first-pass row-only JSONs until the re-sweep
(`run_e18_resweep.sh`, launched on Wolfe's go).

- `sweep_<arm>_<ck>.json`: `lab/divergence/core_depth_sweep.py`, 480 rows, depths
  1,2,3,6,9,12,16, per-row sums. Token files (14 MB each) and probes (40 MB) live under
  `ignored/experiment-artifacts/2026-09-08-arc-e18/`.
- `score_e18.py` / `score_e18.txt`: the depth table, P18a–g, token-level pairing where the
  files exist (labelled), the block-passes cost table.
- `run_<arm>.log`: trainer logs; `queue_e18.log`: the queue log (wall clocks, verdicts).
