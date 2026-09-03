# dmorph — the no-loop MORPH with DiffusionBlocks-routed flow matching

Design: `.agents/notes/rejected/architecture/2026-09-03-dmorph-v1.md`.
Prereg: `lab/experiments/failures/2026-09-03-dmorph-v1-panel.md`.
Branch: `feat/dmorph`, work tree `/home/wolfe/morph-dmorph`.

`research/` holds the five audits (2026-09-03) that pinned the design and the original
handoff note from `feat/db-objective-l2`. Read `2026-09-03-db-testbed-fidelity.md` first
if the question is "was DiffusionBlocks tested properly": yes, byte-exact against the
authors' released modules, and the metric the paper's parity lives on is not perplexity.

Scripts (all read the design note's Implementation record first):

- `run_panel.sh` — the six panel runs (3 arms × 2 seeds), one trainer at a time, GPU
  guard; NOT launched by the build.
- `worth_scorer.py` — the dm-hs four-condition plan worth on a checkpoint over N val
  batches, bootstrap CI, costs against clean.
- `decodability.py` — nearest-neighbour decodability of `x_t` against the embedding
  table on a `t` grid; `t*` and the training mass above it under block-first sampling.

## Results (2026-09-03)

Both preregs failed and both design notes are under `.agents/notes/rejected/architecture/`:
the v1 panel (`lab/experiments/failures/2026-09-03-dmorph-v1-panel.md`) and the v1.1
Fixed-Point-Forcing run (`failures/2026-09-03-dmorph-fpf-tok.md`). Every number, the
checkpoint reads and the row-by-row resolutions:
`lab/experiments/results/2026-09-03-dmorph-v1-panel/README.md`. `score_logs.py` turns the
run logs into the last-N eval means those tables use; `run_panel.sh` takes `DMORPH_ARMS`
(e.g. `dmorph_tok_fpf`) and `DMORPH_SEEDS`.
