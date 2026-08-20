# Agent Note: Public Agent Notes under `.agents/notes`

Status: implemented

## Problem

Decision writeups lived under gitignored `Ai-notes/` (a symlink into the private `ignore/` scratch repo). Agents could not find them, public clones had no rationale trail, and the dated-folder dump mixed plans, run logs, and architecture memos.

## Decision

Decision records are committed at `.agents/notes/{lifecycle}/{class}/yyyy-mm-dd-slug.md`. Format and class set are in [.agents/notes/README.md](../../../README.md). `python scripts/verify_template.py` gates path, Status, and (for new notes) required headings.

Historical `Ai-notes/` markdown was imported in place with an `Origin: Ai-notes/…` line and the original body. Imported files are not format exemplars. New notes must not use `Origin:`.

`docs/` is unchanged. Specs, invariants, and paper archives stay there; Agent Notes link into them. Production code stays in `morph/`. Local artifacts stay in `ignore/`. The `Ai-notes/` / `ai-notes/` symlinks remain gitignored; do not write new notes there.

The nested-dynamics mental model that `CLAUDE.md` cites is
[2026-06-19-iterative-map-dynamics](../architecture/2026-06-19-iterative-map-dynamics.md).

## Alternatives considered

- **Leave notes in `ignore/Ai-notes/`** — keeps the public tree clean but makes every clone and every new agent session blind to decisions.
- **Reorganize `docs/` into cookbook / experiments / postmortem** — useful later; out of scope. Existing specs would break inbound links.
- **Rewrite every imported file into the full skeleton** — would invent `## Alternatives considered` after the fact, which the format forbids.
- **Rename `morph/` to `src/`** — template default for greenfield; MORPH already ships as the `morph` package.

## Consequences

- Public PRs can carry rationale next to code.
- Agents must search `.agents/notes/` before adding a duplicate note.
- Imported notes may still contain stale paths; update facts when touching the decision, or rewrite into the skeleton and drop `Origin:`.
- `ignore/Ai-notes/` still holds a private copy until that scratch repo is cleaned up.
