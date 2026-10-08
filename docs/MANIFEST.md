# MORPH Docs

Top-level navigation for project documentation.

## Architecture & Design

- [Paper references and MORPH usage notes](references.md)
- [MORTAR BCSR + CMS (sparse MLP path)](mortar-bcsr.md)
- [Runtime invariants (BPTT, kernels, compile, phases)](../lab/runtime-invariants.md)
- [Ablation ledger (accepted / rejected / deferred)](ablation-ledger.md)
- [TUL is LXTUL: the 2026-10-07 winner (LXTUL + pointer + DITTO)](../.agents/notes/implemented/architecture/2026-10-07-lxtul-pointer-ditto-winner.md) — the current TUL recipe: `morph/configs/lxtul_pointer.yaml` (5k) then the 1000-step `lxtul_pointer_ditto.yaml` phase; base `lxtul.yaml` ([2026-10-04 note](../.agents/notes/implemented/architecture/2026-10-04-lxtul-primary-candidate.md)); figure [`figures/tul_mechanism.png`](figures/tul_mechanism.png)
- [LXTUL-Parcae testbed snapshot](references/looping-depth/parcae/lxtul-testbed/README.md) — 2026-10 minimal Parcae-based LXTUL testbed (code copy + what it measured)
- [TUL run history: every positive loop-contribution reading (IMPORTANT)](9-26-TUL-run-history-IMPORTANT.md) — 2026-09-22, the 66-row positives ledger; companion of the campaign synthesis note
- [TUL loop contribution: the history](tul-loop-contribution-history.md) — 2026-09-23, every attempt to make the loop contribute, 2026-08-16 to LXTUL-GK: era tables, 144 positives, 48 retractions and qualifications, the docs that still state a retracted number
- [Slot cells: distinct vs blurred](slot-cells-distinct-vs-blurred.md) — 2026-09-27, one table: copies, means and mixture-trained readers blur the cells; WTA + write-all keeps them distinct
- [Parcae TUL proof and MORPH depth-selection plan](../.agents/notes/proposed/architecture/2026-09-05-parcae-tul-depth-transfer-plan.md) includes the completed Parcae CE reference and the new VLD paper reading.
- [TUL — Thought Unpack Loop specification v0.1](../.agents/specs/tul-spec.md) — HISTORY: the v0.1 layout, slot input, loss and generation contract (moved to `.agents/specs/` 2026-10-07); its §3.3 slot-only core and §7 arms were retired 2026-09-03
- [SCSE — Source-Centered State Evolution port specification](../.agents/notes/proposed/architecture/2026-08-25-scse-spec.md) — moved to `.agents/notes/` 2026-10-07
- [TUL-Code — the span code, the tape and the sampler](tul-code-spec.md) — PROPOSED 2026-09-14, not built; the slot holds the code of the span it precedes and the core body samples it by flow matching; decision note `.agents/notes/proposed/architecture/2026-09-14-tul-span-code.md`
- [TUL Gate — span-length and halting gates](../.agents/specs/tul-gate-spec.md) — BUILT 2026-08-22, RETIRED 2026-09-03 with the slot-only core (record; last commit that runs it `d9e04e6`)
- [TUL-FM probing doctrine (flow-matching arc: instruments, controls, phase gates)](../.agents/notes/rejected/architecture/2026-08-28-tul-fm-probing.md) — REJECTED with the FM arc (arc note `.agents/notes/rejected/architecture/2026-08-28-tul-fm-arc.md`)
- [The paid loop: how TUL came to earn its depth, and the recipe that trains it](../.agents/notes/rejected/architecture/2026-09-02-tul-paid-loop-recipe.md) — HISTORY: not TUL (Wolfe, 2026-09-09); LXTUL is the TUL recipe. `base.yaml` still ships the paid loop. Decision note `.agents/notes/implemented/architecture/2026-09-03-ship-the-paid-loop-cut-the-arms.md`
- [The Gist-Slot Recipe — the code that made slot content load-bearing](../.agents/notes/proposed/architecture/2026-08-29-gist-mux-recipe.md) — HISTORY: literate record of GL1b (mask + gradient write + MUX target); retired 2026-09-03; its inversion claim was corrected the same day
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
- [Running offline probes on the 3070, beside the Spark](cookbook/running-probes-on-the-second-host.md)

## TUL satellites (not in this folder)

The current TUL is LXTUL (winner note above). The v0.1 contract is `.agents/specs/tul-spec.md`. Campaign logs and spikes live under
[`lab/tul/`](../lab/tul/). Arm CW / Arm D design notes:

- [Arm CW — compaction window (implemented)](../.agents/notes/archived/architecture/2026-08-18-tul-compaction-window.md)
- [Arm D — teacher distill (proposed)](../.agents/notes/proposed/architecture/2026-08-18-tul-teacher-distill.md)

## Local Archives

- [Reference archive by topic](references/MANIFEST.md)
- [Figure archive by topic](figures/MANIFEST.md) — TikZ/LaTeX architecture diagrams. Measurement plots from wandb are separate: `../lab/experiments/figures/`.
