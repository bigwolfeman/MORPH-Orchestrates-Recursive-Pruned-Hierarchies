# Artifacts: the depth ladder — what core depth ships

Scored 2026-09-19. Record:
[`lab/experiments/failures/2026-09-13-arc-depth-ladder-ship.md`](../../failures/2026-09-13-arc-depth-ladder-ship.md).

The sweep JSONs and their per-token `.npz` files are NOT copied here: they are ~15 MB
each and live in the private scratch tree. Every file below names the source path of
every number it carries.

| File | What |
| --- | --- |
| `pairs_spec.json` | The 20 arm-vs-arm pairs, each naming its two sweep JSONs and the eval depth read from each, plus the bar the record scores it against. This is the input record: re-running the scorer on it reproduces every paired number. |
| `paired_clauses.json` / `.csv` | The scorer's output — CE of each side, the paired delta, and the 95 % bootstrap interval over the 480 validation rows (2,000 draws, seed 0). |
| `depth_curves.csv` | Every arm's whole forced-depth curve at 5k/10k/15k/20k (`ce_d0 … ce_d16`) and its own within-arm K-curve intervals, straight from the sweep JSONs. |
| `rates_and_clock.csv` | Step-200 and step-19800 `tok/s`, max `peak=` over the run, the queue's START/DONE unix epochs, and the step time integrated from the log's `sps` column. |

Reproduce the paired table:

```sh
python lab/divergence/ladder_score.py \
  --spec lab/experiments/results/2026-09-13-arc-depth-ladder/pairs_spec.json \
  --out /tmp/out.json --csv /tmp/out.csv
```

The scorer (`lab/divergence/ladder_score.py`, tests in `tests/test_ladder_score.py`)
refuses to pair two sweeps whose `tok_index` differs. All ten sweeps read here carry the
same 491,520 positions in the same order (480 rows x 1,024 scored positions), checked
before every pair.

Sources, all under `/home/wolfe/morph-scratch/arc/`:

- `results/2026-09-13-depth-ladder/` — d1, d2, d3, d3fixed, d3fixed-s2, p3, p4.
- `results/2026-09-09-norm-match-recipe-reads/` — the top rung `norm-match-20k` at 5k/10k/15k/20k.
- `results/2026-09-14-jobs/` — p4's 5,000-step sweep (re-run by hand after a kill:
  `norm-match-20k-p4/sweep_5000_rerun.log`) and the d6fixed / LoopMTP controls.
- `<arm>/run.log` and `queue.log` — rates, peak memory, wall clock.
