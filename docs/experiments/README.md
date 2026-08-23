# Experiment records

One file per experiment, named `YYYY-MM-DD-slug.md`, filed by outcome. The date is the
date the experiment RAN, not the date the note was edited.

| Folder | Holds |
|---|---|
| `planned/` | Pre-registered question, method and predictions, written BEFORE the run. |
| `results/` | Runs that produced a usable comparison, plus the data behind their figures. |
| `failures/` | Runs that produced no verdict, or whose predictions were falsified. A record here is still a result — it is where the campaign's real cost is written down. |
| `figures/` | Rendered PNGs for the records above. |

A `planned/` file moves to `results/` or `failures/` when it has an answer. It keeps its
original name and its predictions are never edited after the fact; corrections go in a
numbered `## Amendment N` section with the date.

This tree is separate from `.agents/notes/`, which records DECISIONS (why we chose X,
what we gave up). An experiment record says what happened when we measured. If a
measurement changes a decision, the note links the record.

## Figures

`figures/` holds matplotlib PNGs regenerated from wandb, not TikZ diagrams — the
architecture diagrams and their LaTeX sources live in `docs/figures/` and use a
different pipeline (see `docs/figures/MANIFEST.md`). Do not mix the two.

Regenerate everything:

```bash
python scripts/plot_tul_arms.py              # all four
python scripts/plot_tul_arms.py --only ce    # ce | divergence | efficiency | order
```

The script pulls run history from the `morph-tul` wandb project by run id, so a figure
cannot drift from the run it describes. The one exception is the order-parameter probe,
which never went to wandb: its numbers live in `results/tul_order_parameter.csv`, next to
the figure that reads them.

**Colour:** the palette is Okabe-Ito, which is safe for red-green colour blindness, and
no figure uses colour as its only channel — every series also carries its own line style
and marker. Keep it that way.

| Figure | Shows |
|---|---|
| `tul_arms_val_ce.png` | Validation CE for all six arms, plus a tight-axis panel where the 0.007-0.056 nat differences between survivors are actually visible. |
| `tul_arms_divergence.png` | The detonation: CE turning upward, and the gradient norm reaching 3.0e11 while the capped arms stay near 1. |
| `tul_arms_efficiency.png` | Final CE against throughput — the A0 / A1c / A3 trade at equal tokens. |
| `tul_order_parameter.png` | The core-map order parameter across three estimator passes, against the diverged band. |
