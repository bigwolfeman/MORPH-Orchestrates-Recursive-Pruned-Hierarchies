# Depth-candidates panel results (arc row E20)

Record: `../../failures/2026-09-09-arc-e20-loop-depth-candidates.md`. Runner
`morph-scratch/arc/run_e20.sh` at `5c6dec1`. Arms on the Parcae-entry recipe
(`notul_parcae_entry`), one factor each: `depthcand-dense-core` (ternary everywhere but the
looped core), `depthcand-carry-lr20x` (the injection parameters at 20× the base rate),
`depthcand-draw8-bptt4` (Poisson mean 8 / max 12, backprop through the last 4 iterations).

- `sweep_<arm>_<ck>.json`: `lab/divergence/core_depth_sweep.py`, 480 rows, depths
  0,1,2,3,6,8,9,12,16, per-row sums; the per-token files and the grad probes live under
  `ignored/experiment-artifacts/2026-09-09-arc-e20/`.
- `anatomy_<arm>.json`: `core_anatomy.py --rows 3 --depth 8`. `init_probe_<arm>.json`:
  `core_init_probe.py --rows 96`.
- `score_e20.py` / `score_e20.txt`: the E17 rule check, the depth table, P20a–f, the
  token-paired end point (vs `parcae-entry`) and price (vs `plain-depth1`); readers in
  `lab/divergence/sweep_score.py`; references read from `../2026-09-09-arc-e19/`.
- `run_<arm>.log`: trainer logs; `queue_depthcand.log`: the panel's queue lines.
