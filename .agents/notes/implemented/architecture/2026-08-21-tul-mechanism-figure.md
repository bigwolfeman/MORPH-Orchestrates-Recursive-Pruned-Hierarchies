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
