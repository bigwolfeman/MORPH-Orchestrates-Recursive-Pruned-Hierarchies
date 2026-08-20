# Agent Note: Ai-notes migration manifest

Status: implemented

## Problem

The 2026-08-20 migration from private `Ai-notes/` dumps into `.agents/notes/` improved governance, but discoverability became uneven because imported records arrived from multiple source layouts (dated folders, top-level topical notes, and `Incomplete` leftovers). Maintainers need one canonical map to find migrated material quickly without restructuring `docs/`, which is intentionally left unchanged.

## Decision

Adopt this note as the canonical migration manifest for Ai-notes imports. Keep `docs/` intentionally unchanged and use `.agents/notes/` for migration lookup and decision lineage.

| Source group | Destination pattern |
|---|---|
| Dated folders under `Ai-notes/` (for example `Ai-notes/05-31-2026/...`) | `.agents/notes/{lifecycle}/{class}/yyyy-mm-dd-<slug>.md` where date in filename tracks first proposal date when known, otherwise imported date |
| Top-level topical notes in `Ai-notes/` (for example `Ai-notes/diffusionblocks-checklist.md`) | `.agents/notes/{lifecycle}/{class}/yyyy-mm-topic-slug.md` classified by note intent (`architecture`, `process`, `feature`, etc.) |
| `Incomplete/` handling from legacy notes | Keep operational leftovers in gitignored `Incomplete/`; if content is a durable decision, convert into `.agents/notes/{lifecycle}/{class}/...`; if it remains a temporary operator todo, do not import into notes |

Fast lookup for migrated records:

- By class: browse `.agents/notes/{lifecycle}/{class}/` when domain is known (`process`, `architecture`, `testing`, etc.).
- By lifecycle: start at `proposed/`, `implemented/`, `rejected/`, or `archived/` depending on decision status.
- By date: use the `yyyy-mm-dd-` filename prefix to scan chronology quickly.
- By origin marker: search for `Origin: Ai-notes/` to locate imported records that still carry source provenance.

## Alternatives considered

1. Keep migration guidance only in `AGENTS.md`.
   - Rejected because root standing orders should stay short and not become a migration catalog.
2. Build a generated global index file for all migrated notes.
   - Rejected because `.agents/notes/README.md` explicitly avoids adding an index and favors folder+search navigation.
3. Move migration guidance into `docs/`.
   - Rejected because `docs/` is intentionally left unchanged for this migration and remains reserved for specs/invariants/paper notes, not decision-record logistics.

## Consequences

Discoverability improves with one stable mapping reference and minimal cross-links from root guidance. Future imports remain consistent with lifecycle/class/date naming, while preserving the policy that `docs/` stays structurally unchanged. Imported legacy notes remain traceable until rewritten into standard Agent Note format.
