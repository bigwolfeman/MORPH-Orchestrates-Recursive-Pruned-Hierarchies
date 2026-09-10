# Slot-map levers panel + the reread: results

Records: `../../failures/2026-09-10-arc-slot-map-levers.md`,
`../../failures/2026-09-10-arc-slot-reread.md`. Runner `morph-scratch/arc/run_slotloop3.sh`
(file-driven queue), arms pinned per line: the levers at `1edd4c0`, the noise-entry re-run
at `98fd568`, the reread at `13b1cbc`. All arms are `tul_slot_unpack_norm_match` plus one
change; the reference readouts of that arm live in `../2026-09-09-slot-loop-norm-match/`.

- `sweep_<arm>_<ck>.json`: `core_depth_sweep.py`, 480 rows, forced slot depths
  1,2,3,6,9,12,16, token and forecast (`mux_local`) curves with bootstrap CIs. The
  per-token arrays (`*.tokens.npz`) and the grad probes (`probe_<arm>.jsonl`) are under
  `ignored/experiment-artifacts/2026-09-10-slot-map-levers/`.
- `worth_<arm>_5000.json`: `worth_profile.py` (zero / shuffle / wrong_seed by offset bin).
- `slot_state_<arm>_5000.json`: `slot_state_probe.py` (the looped state per forced depth).
  INVALID for `slot-unpack-fixed-depth`: the fixed knob overrides the probe's forcing.
- `slot_anatomy_<arm>_5000.json`: `slot_anatomy.py` (new): per pass at forced depth 8 over
  635 valid slots, the state's norm and rank, its movement (stream mean and full carrier),
  every core block's attention / MLP branch out-in ratio, the ctx channel's norm share, and
  for the reread arm the read term over the state. Also run on the three 2026-09-09 slot
  arms (`slot-unpack-norm-match`, `slot-mux-norm-match`, `slot-loop-norm-match`).
- `sweep_plain-panel-norm-match_*.json`, `anatomy_plain-panel-norm-match_5000.json`,
  `init_probe_plain-panel-norm-match_5000.json`: the plain ruler (prelude entry, norm_match)
  from the slot-loop panel, copied here because every reading above is read against it.
- `run_<arm>.log`: trainer logs. `score_levers.py`: the scorer (rerun from the repo root
  with `PYTHONPATH=. python lab/experiments/results/2026-09-10-slot-map-levers/score_levers.py`).
