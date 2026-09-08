# ARC E16 results — the Olympiad curriculum panel (2026-09-08)

Record: `lab/experiments/failures/2026-09-08-arc-e16-olympiad-curriculum-panel.md`.

- `sweeps/` — the runner's depth sweeps (`lab/divergence/olympiad_sweep.py`, 1000 docs drawn from
  the ORIGINAL `eval_holdout.jsonl` files, 41 % of whose docs sit verbatim in the training bands).
- `sweeps_clean/` — the same 16 checkpoints on all 1,906 docs of `olympiad_bands/holdout_clean/`
  (absent from every training band view, unique within the holdout). This set decides.
- `run_<arm>.log` — training logs (the `[VAL n]` lines are the CONTAMINATED held-out val).
- `score_e16.py` → `score_e16.txt` — P16a–j tables, per-band accuracy, paired between-arm CE.
- `PROBES.md` — where the 15–20 MB per-step probe JSONL files live (gitignored).
