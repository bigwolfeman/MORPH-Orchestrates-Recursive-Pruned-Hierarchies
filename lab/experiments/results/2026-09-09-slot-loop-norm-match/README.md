# Slot-loop panel under norm_match: results

Record: `../../failures/2026-09-09-arc-slot-loop-norm-match.md`. Arms `slot-loop-norm-match`,
`slot-loop-absmean` (A1, no MUX), `slot-mux-norm-match`, `slot-mux-absmean` (M-next), the
plain ruler `plain-panel-norm-match`, and the coda-reads-the-thought arm
`slot-unpack-norm-match` (`../../failures/2026-09-09-arc-coda-reads-the-thought.md` once
filed). Runners `morph-scratch/arc/run_slotloop.sh` → `run_slotloop2.sh` → `run_slotloop3.sh`.

- `sweep_<arm>_<ck>.json`: `core_depth_sweep.py` (slot arms: forced slot depths 1..16, token
  and forecast curves; the plain ruler: depths 0..16). Per-token arrays and grad probes under
  `ignored/experiment-artifacts/2026-09-09-slot-loop-norm-match/`.
- `worth_<arm>_5000.json`: `worth_profile.py`. `slot_state_<arm>_<ck>.json`: `slot_state_probe.py`.
- `anatomy_plain-panel-norm-match_5000.json`, `init_probe_plain-panel-norm-match_5000.json`:
  the ruler's core anatomy and entry probe. The slot arms' anatomies (`slot_anatomy.py`) are
  in `../2026-09-10-slot-map-levers/`.
- `run_<arm>.log`: trainer logs.
