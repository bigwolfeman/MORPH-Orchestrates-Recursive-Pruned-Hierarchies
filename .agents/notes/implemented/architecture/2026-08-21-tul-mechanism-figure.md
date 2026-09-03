# Agent Note: TUL mechanism figure replaces stack-dump overview

Status: implemented

## Problem

`docs/figures/tul_overview.png` restated MORPH internals (attention alphabet,
HC/GLA, head pipeline, paper citations) already covered by
`morph_overview.png`. It hid the actual TUL story: one core loop per thought,
freeze that state in the shared sequence, and AR-decode the next span until the
punctuation / cap cut. Readers could not tell that `z` is kept, or that v1 has
no learned exit gate.

## Decision

Ship `docs/figures/architecture/tul_mechanism.tex` → `docs/figures/tul_mechanism.png`
as the TUL diagram. Top-down: input span strip → prelude → think (core × T on
slots, save `z`) → freeze callout → decode → output span strip with `z` kept.
Deprecate `tul_overview.tex` as `tul_overview_deprecated.tex`. Point README and
`CLAUDE.md` at the new preview. States are named `z`, matching the layout
slots, not `h`.

## Alternatives considered

- **Edit in place on `tul_overview.tex`:** would keep a misleading filename and
  leave stale PNG references easy to miss. Rejected in favor of a new document
  plus an explicit deprecate rename.
- **Keep dual-path “TUL gate” framing (tokens skip / slots gather) as the
  centerpiece:** factually part of the forward, but not the mental model in
  README “TUL For Dummies”; it overweighted plumbing vs amortize-loop-over-span.
- **Draw a learned exit / length-predict gate:** not in this tree (spec §3.5
  “learned per-slot exit … later”; §6 “exit head later”). Showing it would be
  false for v1.

## Consequences

- Regenerating the README figure uses `tul_mechanism.tex` and
  `pdftoppm … ../tul_mechanism` from `docs/figures/architecture/` (MANIFEST path
  corrected from `../../`).
- Stale `tul_overview.png` may linger until deleted; do not link it.
- Spec text in `docs/tul-spec.md` is unchanged; the figure now matches the
  dummies summary and the implemented forward (`_tul_core` → `prefix_project` →
  coda with `coda_sees_slots=true`).

## Update 2026-09-03 — redrawn for the paid loop

The 2026-08-21 figure drew the slot-only loop (core on slot positions, freeze `z` as an
attended prefix, a separate decode loop over the next span, slot input = span mean). That
forward was cut on 2026-09-03 (`2026-09-03-ship-the-paid-loop-cut-the-arms.md`). The figure
now draws the shipped paid loop: tokens and slots are ONE row (two positions per slot),
prelude, the per-sample Poisson-depth core (full BPTT) and the coda all run on every
position, nothing is gathered or scattered, and the DECODE box states the recipe in one line (every
token label at `plast_weight` 1.0, the slot's own emit label at `emit_weight` 0, the next
span's first token read at the boundary token, full recompute per generation step). The
slot input is the boundary seed `E_slot + W_sent · embed(t_last)`. The "z kept in the row"
callout stays. `morph_overview.tex` was corrected in the same change from "truncated BPTT
(last 4 iters)" to full BPTT (`bptt_depth` 8 ≥ max); its GLA branch is drawn as before
even though `base.yaml` runs `retention: false` (Wolfe: leave GLA in, Raven attention may
take its place). Still not fixed there: the core is labelled ×2N layers against a prelude
of N; the recipe is 4:6:4 locally and 4:8:4 on the cloud target.
Second pass the same day, after Wolfe's read ("less clear now"): the think / keep /
decode narrative and the generation feedback loop are back. THINK is the core on the
whole row with "the loop's output at the slot cells is the thought z"; the keep callout
says every later position attends to z in the core and the coda; DECODE is the coda on
all positions with the head emitting d, and the emitted row feeds back into the PRELUDE
(the whole row is run again; a boundary emitted inserts two slot cells after it). The CE
statement is one clause inside DECODE, not a separate box.
