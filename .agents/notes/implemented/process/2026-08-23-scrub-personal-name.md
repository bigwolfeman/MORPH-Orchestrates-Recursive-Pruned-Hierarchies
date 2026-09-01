# Agent Note: Remove personal name from docs and notes

Status: implemented

## Problem

Public specs, configs, code comments, and active Agent Notes attributed
decisions to a personal name. That name does not belong in the tree.

## Decision

Scrub person-name attributions from:

- `docs/` (esp. `tul-spec.md`, `ablation-ledger.md`); redefine **[W]** as
  “project design decision”
- `morph/` comments and YAML comment prose (leave `/home/…` dataset paths)
- active `.agents/notes/{proposed,implemented,rejected}/`
- `tile-prover/journal.md`

Do **not** edit `.agents/notes/archived/` (frozen). Do not rewrite paper dumps
under `docs/references/`. Do not rename hub ids or local cache paths that
happen to contain the same string as a filesystem username.

## Alternatives considered

- Scrub archives too — rejected; archived notes are sealed.
- Replace paths `/home/…` — rejected; would break local runbooks without a
  portable substitute in this change.

## Consequences

New writing must not reintroduce personal names. Attribute findings to MORPH /
measurements / papers.
