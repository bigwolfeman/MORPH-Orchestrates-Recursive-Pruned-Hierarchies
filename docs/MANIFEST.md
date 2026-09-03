# MORPH Docs

Top-level navigation for project documentation.

## Architecture & Design

- [Paper references and MORPH usage notes](references.md)
- [MORTAR BCSR + CMS (sparse MLP path)](mortar-bcsr.md)
- [Runtime invariants (BPTT, kernels, compile, phases)](../lab/runtime-invariants.md)
- [Ablation ledger (accepted / rejected / deferred)](ablation-ledger.md)
- [TUL — Thought Unpack Loop specification](tul-spec.md) — layout, slot input, loss and generation contract; its §3.3 slot-only core and §7 arms are RETIRED 2026-09-03 (the paid loop below is the shipped forward)
- [SCSE — Source-Centered State Evolution port specification](scse-spec.md)
- [TUL Gate — span-length and halting gates](tul-gate-spec.md) — BUILT 2026-08-22, RETIRED 2026-09-03 with the slot-only core (record; last commit that runs it `d9e04e6`)
- [TUL-FM probing doctrine (flow-matching arc: instruments, controls, phase gates)](tul-fm-probing.md) — PROPOSED 2026-08-28; arc note in `.agents/notes/proposed/architecture/`
- [The paid loop: how TUL came to earn its depth, and the recipe that trains it](tul-paid-loop-recipe.md) — THE SHIPPED FORWARD since 2026-09-03 (`base.yaml`, master); §6 is the recipe and names the unmeasured conjunction; decision note `.agents/notes/implemented/architecture/2026-09-03-ship-the-paid-loop-cut-the-arms.md`
- [The Gist-Slot Recipe — the code that made slot content load-bearing](gist-mux-recipe.md) — LIVE 2026-08-29; literate record of GL1b (mask + gradient write + MUX target), the arm that inverted the mask's price
- [Known-good runs and environment assumptions](../.agents/notes/implemented/process/2026-07-03-known-good-runs.md)
- [Data placement design spec](../.agents/notes/implemented/architecture/2026-07-03-data-placement-design.md)
- [MORPH / Olympiad-AI interop contract](olympiad-interop.md)

## Experiment records

What happened when we measured, one file per experiment, filed by outcome. Decisions
live in `.agents/notes/`; these are the runs behind them.

- [Experiment records — layout, naming and figure regeneration](../lab/experiments/README.md)
- [TUL arms — the first complete comparison](../lab/tul/arms-result.md) — A0 / A0c / A1c / A3 at 20k steps
- [The gated-TUL bake-off](../lab/experiments/failures/2026-08-21-tul-gate-bakeoff.md) — no verdict; every arm died
- [What makes the TUL arms diverge?](../lab/experiments/failures/2026-08-22-tul-divergence-cause.md)
- [Is `ademamix_alpha_cap=1.0` a cure or a delay?](../lab/experiments/failures/2026-08-22-tul-order-parameter.md)
- [Root cause of the TUL core takeover](../lab/experiments/results/2026-08-24-tul-takeover-rca.md) — the per-block backward gain, and what does not stop it
- [The takeover is a forward state collapse](../lab/experiments/failures/2026-08-24-tul-takeover-cure.md) — four weight-space cures fail; per-slot input embeddings double the time to failure and improve CE on both seeds, and still do not cure

## Cookbook

Step-by-step procedures. How to do a thing, not why it is done that way.

- [Replaying the TUL core takeover from a checkpoint](cookbook/replaying-the-core-takeover.md)
- [Measuring the looped core's operator, not its magnitudes](cookbook/measuring-the-core-map.md)

## TUL satellites (not in this folder)

Canonical contract stays here as `tul-spec.md`. Campaign logs and spikes live under
[`lab/tul/`](../lab/tul/). Arm CW / Arm D design notes:

- [Arm CW — compaction window (implemented)](../.agents/notes/archived/architecture/2026-08-18-tul-compaction-window.md)
- [Arm D — teacher distill (proposed)](../.agents/notes/proposed/architecture/2026-08-18-tul-teacher-distill.md)

## Local Archives

- [Reference archive by topic](references/MANIFEST.md)
- [Figure archive by topic](figures/MANIFEST.md) — TikZ/LaTeX architecture diagrams. Measurement plots from wandb are separate: `../lab/experiments/figures/`.
