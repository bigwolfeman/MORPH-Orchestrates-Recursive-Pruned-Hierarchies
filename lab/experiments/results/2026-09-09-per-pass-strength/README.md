# Per-pass-strength panel results

Record: `../../successes/2026-09-09-arc-per-pass-strength.md`. Runner
`morph-scratch/arc/run_strength.sh` at `d2670e2` (`NOWAIT=1`). Arms on the Parcae-entry
recipe (`notul_parcae_entry`), one factor each: `scale-norm-match`
(`ternary_scale_mode: norm_match`, the ternary weight keeps the latent's norm), `scale-ttq`
(learnable γ₊ / γ₋ per tensor), `threshold-03` (`ternary_threshold: 0.3`).
`precision-bf16-all` (`training.ternary: false`, the ceiling) has not run yet; the scorer
prints `NOT RUN YET` for it and picks it up from the same file names when it does.

- `sweep_<arm>_<ck>.json`: `lab/divergence/core_depth_sweep.py`, 480 rows, depths
  0,1,2,3,6,9,12,16, per-row sums; the per-token files (`*.tokens.npz`) and the grad probes
  (`probe_<arm>.jsonl`) live under `ignored/experiment-artifacts/2026-09-09-per-pass-strength/`.
- `anatomy_<arm>.json`: `core_anatomy.py --rows 3 --depth 8` (the branch out/in per block
  at iteration 0 and 6, the state movement per pass, the carrier norm and rank).
  `init_probe_<arm>.json`: `core_init_probe.py --rows 96`.
- `score_strength.py` / `score_strength.txt`: the E17 rule check, the depth table, the
  mechanism (P-str-b), the contribution (P-str-c), the Binding check, the horizon readings
  against `parcae-entry` (`../2026-09-09-arc-e19/`) and `depthcand-dense-core`
  (`../2026-09-09-arc-e20/`), the TTQ learned-scale readout off
  `checkpoints/morph/scale-ttq/step_5000.pt`, the init probes, survival and cost. Readers in
  `lab/divergence/sweep_score.py`.
- `run_<arm>.log`: trainer logs; `queue_strength.log`: the panel's queue lines.

Rerun from the repo root:

```
PYTHONPATH=. python lab/experiments/results/2026-09-09-per-pass-strength/score_strength.py
```
