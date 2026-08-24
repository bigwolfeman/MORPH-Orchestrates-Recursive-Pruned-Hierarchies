# MORPH Docs

Top-level navigation for project documentation.

## Architecture & Design

- [Paper references and MORPH usage notes](references.md)
- [Runtime invariants (BPTT, kernels, compile, phases)](runtime-invariants.md)
- [Ablation ledger (accepted / rejected / deferred)](ablation-ledger.md)
- [TUL — Thought Unpack Loop specification (`experiments/tul`)](tul-spec.md)
- [TUL core-takeover campaign — RUNNING INDEX of every hypothesis tried](tul-takeover-campaign.md)
- [TUL Gate — span-length and halting gates](tul-gate-spec.md) — BUILT 2026-08-22; §13 lists what building it changed in the spec
- [Known-good runs and environment assumptions](../.agents/notes/implemented/process/2026-07-03-known-good-runs.md)
- [Data placement design spec](../.agents/notes/implemented/architecture/2026-07-03-data-placement-design.md)
- [MORPH / Olympiad-AI interop contract](olympiad-interop.md)

## Experiment records

What happened when we measured, one file per experiment, filed by outcome. Decisions
live in `.agents/notes/`; these are the runs behind them.

- [Experiment records — layout, naming and figure regeneration](experiments/README.md)
- [TUL arms — the first complete comparison](experiments/results/2026-08-18-tul-arms-first-comparison.md) — A0 / A0c / A1c / A3 at 20k steps
- [The gated-TUL bake-off](experiments/failures/2026-08-21-tul-gate-bakeoff.md) — no verdict; every arm died
- [What makes the TUL arms diverge?](experiments/failures/2026-08-22-tul-divergence-cause.md)
- [Is `ademamix_alpha_cap=1.0` a cure or a delay?](experiments/failures/2026-08-22-tul-order-parameter.md)
- [Root cause of the TUL core takeover](experiments/results/2026-08-24-tul-takeover-rca.md) — the per-block backward gain, and what does not stop it
- [The takeover is a forward state collapse](experiments/failures/2026-08-24-tul-takeover-cure.md) — four weight-space cures fail; per-slot input embeddings double the time to failure and improve CE on both seeds, and still do not cure

## Cookbook

Step-by-step procedures. How to do a thing, not why it is done that way.

- [Replaying the TUL core takeover from a checkpoint](cookbook/replaying-the-core-takeover.md)
- [Measuring the looped core's operator, not its magnitudes](cookbook/measuring-the-core-map.md)

## Local Archives

- [Reference archive by topic](references/MANIFEST.md)
- [Figure archive by topic](figures/MANIFEST.md) — TikZ/LaTeX architecture diagrams. Measurement plots from wandb are separate: `experiments/figures/`.
