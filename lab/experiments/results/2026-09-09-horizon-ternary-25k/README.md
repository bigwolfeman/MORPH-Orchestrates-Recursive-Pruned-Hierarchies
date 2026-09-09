# Horizon run results (stopped at 10k) and the old entry's 20k curve

Record: `../../failures/2026-09-09-arc-horizon-ternary-25k.md`. Runner `morph-scratch/arc/run_old20k_then_horizon.sh`
at `6c98f8e` (the old control's sweeps first, then `run_horizon.sh`, stopped by hand at step 10,200).

- `old_entry_20k/sweep_old-entry-notul-20k_<ck>.json` (ck 2500..20000): `core_depth_sweep.py` on the
  surviving 20k plain control (`morph-scratch/checkpoints-keep/notul-20k-wu`, config `notul`), 480 rows,
  depths 0,1,2,3,6,9,12,16; `anatomy_old-entry-notul-20k_{5000,10000,20000}.json`.
- `sweep_horizon-ternary-25k_{5000,10000}.json`, `anatomy_horizon-ternary-25k_{5000,10000}.json`: the
  Parcae-entry recipe's 25k run (`notul_horizon_ternary_25k`) up to its stop.
- Token files and probes under `ignored/experiment-artifacts/2026-09-09-horizon-ternary-25k/`.
- `run_horizon-ternary-25k.log`, `queue_horizon.log`.
