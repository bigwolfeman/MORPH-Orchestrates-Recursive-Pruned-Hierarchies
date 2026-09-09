# Parcae-entry panel results (arc row E19)

Record: `../../failures/2026-09-09-arc-e19-parcae-loop-entry.md`. Runner
`morph-scratch/arc/run_e19.sh` at `3ed1ea2`. Arms (renamed at filing from their index
names): `parcae-entry` (the plain model with Parcae's loop entry: noise state init, all-dim
carry with learned B, no fixed-point term), `plain-no-fixed-point` (the plain model with the
fixed-point term off), `plain-depth1` (the plain model trained at depth 1, the price control).

- `sweep_<arm>_<ck>.json`: `lab/divergence/core_depth_sweep.py`, 480 rows, depths
  0,1,2,3,6,9,12,16, per-row sums; the per-token files (14 MB each) and the grad probes
  live under `ignored/experiment-artifacts/2026-09-09-arc-e19/`.
- `anatomy_<arm>.json`: `core_anatomy.py --rows 3 --depth 8`. `init_probe_<arm>.json`:
  `core_init_probe.py --rows 96`.
- `score_e19.py` / `score_e19.txt`: the E17 rule check, the depth table, P19a–g, the
  token-paired end point and price (readers in `lab/divergence/sweep_score.py`).
- `run_<arm>.log`: trainer logs; `queue_parcae_entry.log`: the panel's queue lines.
