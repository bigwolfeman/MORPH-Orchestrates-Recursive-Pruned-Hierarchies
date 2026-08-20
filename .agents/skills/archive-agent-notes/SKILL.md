---
name: archive-agent-notes
description: Use when adding, auditing, archiving, or deleting Agent Notes. Checks new notes for superseded records and moves completed implemented notes to archived/{class}/.
---

# Archive Agent Notes

Read [.agents/notes/README.md](../../notes/README.md) and [.agents/notes/archived/AGENTS.md](../../notes/archived/AGENTS.md) first.

## When adding a note

Search `.agents/notes/proposed`, `implemented`, and `rejected` for the same decision. Archive fully superseded implemented notes in the same change. Cross-link partial supersessions. Reject obsolete proposals. Delete rejected notes that no longer prevent a real mistake.

## Classify implemented notes

Keep active when rationale, alternatives, negative guarantees, ownership, security, or reintroduction conditions still guide work.

Archive when the decision is complete and the body is unlikely to guide future work (one-off chrome, closed minor bugs, process history now obvious elsewhere).

Never archive a proposed note. Reject it instead.

## Archive steps

1. Move `implemented/{class}/yyyy-mm-dd-slug.md` to `archived/{class}/yyyy-mm-dd-slug.md`.
2. Insert `Archived: YYYY-MM-DD` immediately under `Status: implemented`. No other body edits.
3. Repair inbound links from active docs and from `CLAUDE.md`.
4. Run `python scripts/verify_template.py`.
