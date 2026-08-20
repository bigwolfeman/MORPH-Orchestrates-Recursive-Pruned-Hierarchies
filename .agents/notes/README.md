# Agent Notes

One kind of design doc lives here. An **Agent Note** records a decision or proposal that affects this codebase: the *why*, *what we gave up*, and how to verify. It is not a run log, not a postmortem, and not a substitute for `docs/`.

Human-facing specs, invariants, and paper notes stay in `docs/` — do not restructure that tree to match this one. Link to those files.
- Migration map for imported `Ai-notes/` records: [implemented/process/2026-08-20-ai-notes-migration-manifest.md](implemented/process/2026-08-20-ai-notes-migration-manifest.md).

## Layout and naming

Path: `{lifecycle}/{class}/yyyy-mm-dd-topic-title.md`

**Lifecycle** (status). Move the file when status changes; update `Status:` in the same change.

| Folder | Meaning |
|---|---|
| `proposed/` | Not built, or only partly built |
| `implemented/` | Shipped. Keep facts (paths, names) current with the code. Do not rewrite the decision. |
| `rejected/` | Considered and declined. Keep only while the rationale still prevents a tempting mistake; otherwise delete. |
| `archived/` | Frozen implemented notes that no longer guide work. Never edit after sealing. Path is `archived/{class}/…` (no `implemented/` segment). |

The date is when the topic was **first proposed**. Cross-links use relative Markdown paths.

Do not add an `INDEX.md`. Browse folders or search.

## Classification

Closed set. Adding a class requires an Agent Note and an update to `scripts/verify_template.py`.

| Class | Covers |
|---|---|
| `feature` | New user- or model-facing capability |
| `bug-fix` | Defect or a gap a postmortem surfaced |
| `simplification` | Removes code or surface without adding a capability |
| `architecture` | Structure of shipped source |
| `process` | Tooling, policy, workflow around the code |
| `testing` | Test infrastructure and strategy |

`architecture` is shipped source. `process` is surrounding workflow. There is no `refactor` class; use `simplification` or `architecture`.

## When to write one

Every non-trivial change adds or updates at least one Agent Note in the same change. Non-trivial: behavior, architecture, shared contracts, process, testing strategy, on-disk or wire formats, or any decision a maintainer may revisit.

Update the owning note; do not duplicate. A different decision needs a new note, cross-linked. Only mechanical or local edits are exempt.

A proposal for future work starts in `proposed/`. A decision already made starts in `implemented/`.

## File format (new notes)

First lines, exactly:

```markdown
# Agent Note: <title>

Status: <status>
```

Then a blank line. `Status:` must match the folder:

- `Status: proposed`
- `Status: implemented`
- `Status: rejected — <why, in one line>`

### `proposed/`

```markdown
## Problem
## Proposal
## Alternatives considered
## Acceptance criteria
## Risks
```

### `implemented/`

```markdown
## Problem
## Decision
## Alternatives considered
## Consequences
```

Present tense. No `## Proposal`, `## Plan`, `## Migration plan`, or `## Acceptance criteria`.

### `rejected/`

Keep the proposal-time sections. The verdict is the `Status:` line. Must include `## Problem`, `## Proposal`, and `## Alternatives considered`.

### Alternatives considered — mandatory

Each genuine alternative and why it lost. Record alternatives; do not invent them after the fact.

### Moves

`proposed/` → `implemented/`: rewrite Proposal into present-tense Decision; fold Acceptance criteria and Risks into Consequences or a present-tense Testing section.

`proposed/` → `rejected/`: add the reason on `Status:` and freeze.

## Imported notes (from `Ai-notes/`)

Files that contain `Origin: Ai-notes/` were copied from the private dated-folder dump on 2026-08-20. They keep their original body. Do not treat them as format exemplars. Do not add new files with an `Origin:` line — write a real Agent Note instead. If you update an imported note's *decision*, rewrite it into the skeleton above and drop `Origin:`.

## Where not to write

- Not `Ai-notes/` or `ai-notes/` (gitignored private scratch).
- Not `ignore/` for anything that should be in git.
- Not `docs/` for decision rationale (link the spec; put the *why* here).

## Archiving

Archive an implemented note when it is complete and unlikely to guide future work. Keep it active when alternatives, negative guarantees, ownership, security, or reintroduction conditions still matter.

Archival: move the file, insert `Archived: YYYY-MM-DD` immediately under `Status: implemented`, repair inbound links. Then never edit it. Use [.agents/skills/archive-agent-notes/SKILL.md](../skills/archive-agent-notes/SKILL.md).
