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
python scripts/plot_tul_arms.py --only ce    # ce | divergence | efficiency | order | bakeoff | decode
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
| `tul_decode_modes.png` | rep4 by decode mode for every arm, with CE beside each name and a real-text line. The figure that exists because one decode mode hid a greedy repetition loop. |
| `tul_bakeoff.png` | The gate bake-off live: val CE, the degeneration watch (rep4 / distinct3), and the halting policy's mean chosen depth. Arms appear as they start; stale wandb ids are dropped by name and listed in the caption. |
| `tul_order_parameter.png` | The core-map order parameter across three estimator passes, against the diverged band. |


## Generation samples

`val CE cannot see degeneration.` A repetition loop scores an excellent perplexity — a
measured 1.46 against real text's 32.44 — so every fluency number needs a diversity
number beside it. `rep4` and `distinct3` are that pair and are logged by the training
loop as `gen/*`.

The training loop's own generation test decodes at **one** setting, `temperature 0.8 /
top-k 50`. That is not enough. Greedy is where a repetition loop actually appears, and
pure ancestral sampling is where a collapsed readout shows up as "sampling is identical
to greedy". For the full table:

```bash
python scripts/tul_samples.py \
  --ckpt gate_20k=tul_gate=checkpoints/morph/tul-gate/step_20000.pt --halt
```

It scores greedy, top-k 50 at t=0.8, and pure ancestral t=1.0, each with rep4 /
distinct3 / span geometry, plus a REAL TEXT row scored by the same code as the anchor.
Rank nothing against a row whose distinct3 is far from the real-text value.

**Run it on finished checkpoints, not during a campaign.** `gate_bakeoff.sh` launches
each arm as its own process, so editing the training loop mid-campaign makes later arms
differ from earlier ones by more than the variable under test — and a second model on
the GPU can OOM a training arm that is already sitting on ~3.8 GB of margin.
