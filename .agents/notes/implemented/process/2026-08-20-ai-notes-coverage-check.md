# Agent Note: Ai-notes Coverage Check

Status: implemented

## Problem

The repository needed a deterministic check that markdown content under
`ignore/Ai-notes` was migrated into `.agents/notes` via imported notes that
declare source lineage with `Origin: Ai-notes/...`.

## Decision

Run a path-normalized coverage check with these rules:

- Source set: all `*.md` under `ignore/Ai-notes`.
- Explicit exclusions: `ignore/Ai-notes/AGENTS.md` and `ignore/Ai-notes/README.md`.
- Destination index: all lines in `.agents/notes/**/*.md` that start with
  `Origin: Ai-notes/`.
- Normalization: treat origin paths as relative to `ignore/Ai-notes` and compare
  normalized POSIX-style relative paths.

Result on 2026-08-20:

- Source markdown files considered: 52
- Unique normalized `Origin:` mappings found: 52
- Missing source files: 0
- Extra origin mappings: 0
- Duplicate origin mappings: 0

No unmigrated Ai-notes markdown remains under the defined source set.

## Alternatives considered

- Manual eyeballing of both trees: rejected because it is error-prone and not
  reproducible.
- Filename-only comparison without normalization: rejected because nested dated
  paths require consistent relative-path mapping.

## Consequences

The migration coverage is verified as complete for the scoped source set with
clear, repeatable matching criteria and explicit exclusions.
