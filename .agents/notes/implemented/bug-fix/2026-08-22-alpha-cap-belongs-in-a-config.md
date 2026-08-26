# Agent Note: the TUL divergence fix lived in a shell script, not a config

Status: implemented

## Problem

The 2026-08-21 gated-TUL bake-off ran three arms. Two diverged. The divergence had
already been diagnosed and fixed on 2026-08-17, and two arms had already been carried to
20000 steps under that fix. The bake-off still detonated, because **the fix was never
written to a config file**.

`training.ademamix_alpha_cap = 1.0` existed exactly once in the tree: as

    CAP="training.ademamix_alpha_cap=1.0"

on line 47 of `00-MORPH-TUL/ignore/tul_logs/run_tul_arms2.sh` — a **gitignored** script in
a private scratch repo. The commit that adopted the fix, `4650cb1`, changed 36 lines of
one markdown file and zero lines of YAML. `gate_bakeoff.sh` passes no optimiser
overrides, so all three bake-off arms silently inherited `base.yaml`'s
`ademamix_alpha_cap: 3.5` — the pre-fix recipe.

A fix that only exists in an untracked launcher is not a fix. It is a fact about one
person's shell history, and it expires the moment a different script launches the run.

## Decision

`ademamix_alpha_cap: 1.0` goes in `morph/configs/tul_short.yaml`, the file every TUL arm
inherits, with the evidence recorded inline:

- control diverged 5/5 (aborts at steps 2080, 3240, 4540, 5900, 6200)
- placebo diverged 2/2 — survival under the cap is not free trajectory luck
- the cap survived 2/2 at 4800, then both arms to 20000
  (`tul-a0-acap1` 3.2805, `tul-a1-acap1` 3.2243)

It goes in `tul_short.yaml` and **not** in an arm file, because it is an optimiser
setting. Putting it in `tul_a1.yaml` would make the A1-vs-A0 comparison differ by two
things at once, which is the confound the arms exist to avoid.

`base.yaml` keeps 3.5 and is untouched. The cap was only ever measured on the TUL
recipe; extending it to the 100k-step production schedule is a claim nobody has tested.

Verified by Hydra compose: `base` resolves to `batch=4, alpha_cap=3.5`;
`tul_short`, `tul_a0`, `tul_a1`, `tul_a1r`, `tul_a3`, `tul_gate` all resolve to
`batch=12, alpha_cap=1.0, grad_clip=1.0, steps=20000`.

## Alternatives considered

- **Put the cap in `base.yaml`.** Rejected. It would apply the cap to the production
  100k schedule, where it has never been measured, in order to fix a 20k TUL problem.
- **Keep passing it as a launcher override, and fix the launcher.** Rejected. That is the
  same failure one layer up: the next script written by the next session inherits nothing.
  The config is the artifact wandb records, so the config is where a recipe belongs.
- **Put it in each arm file.** Rejected. Six copies drift, and it makes the arms differ
  by an optimiser setting on top of the variable under test.

## Consequences

- Every TUL arm now carries the cap by construction. A run cannot silently omit it.
- wandb config alone is enough to tell whether a past TUL run had the fix. Before this
  change it was not, which is exactly why the bake-off's divergence was mysterious.
- The cap remains **verified at batch 14 only**. `tul_short.yaml` now sets batch 12, which
  changes the gradient noise scale that drives the slow EMA the cap throttles. That gap is
  the subject of `lab/experiments/planned/2026-08-22-tul-order-parameter.md`.
- A broader lesson, worth its own check: a fix is adopted when it lands in a **tracked**
  file that the run path reads. A markdown note recording a fix is documentation, not
  adoption.
