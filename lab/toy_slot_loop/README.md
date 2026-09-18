# Toy slot-loop study

A small PyTorch model of MORPH's slot TUL, used to ask which loss attachments and which
carries make a shared-weight slot loop earn depth, on tasks whose iteration requirement is
a known property of the data.

- [PLAN.md](PLAN.md) — the grid and the predictions, frozen before the runs.
- [WRITEUP.md](WRITEUP.md) — the report.
- `model.py` `tasks.py` `instruments.py` — the model, the tasks, the probes.
- `run_cell.py` — trains and instruments one cell, writes one JSON.
- `aggregate.py` — turns the JSONs into the tables in the write-up.
- `selfcheck.py` — the contracts. Run it before trusting a number.

```
python selfcheck.py
python run_cell.py --task compose --attach exit --geometry strict --seed 0 --out cell.json
python aggregate.py <dir-of-jsons>
```

Result JSONs are not committed; they live under
`ignored/experiment-artifacts/2026-09-10-toy-slot-loop/`.

## 2026-09-18: the `eliminate` task

A third task, added for the superposition question. `compose` and `summary` are untouched
and bit-identical. `eliminate` makes the optimal intermediate state a SET of candidates that
later evidence prunes, not a point, and it carries three extra instruments
(`candidate_mass`, `membership_probe`, `twin_divergence`) plus its own enumerated ceilings
(`tasks.eliminate_ceilings`). Pre-registration:
[`lab/experiments/planned/2026-09-18-toy-eliminate-deferred-commitment.md`](../experiments/planned/2026-09-18-toy-eliminate-deferred-commitment.md).

```
python run_cell.py --task eliminate --attach staged --seed 0 --steps 4000 --out cell.json
```
