# Agent Note: Public Agent Notes Import Gaps
Status: rejected — historical scratch imported from private Incomplete
Origin: Ai-notes/Incomplete/2026-08-20-public-agent-notes.md
Imported: 2026-08-20 from private Incomplete import pass

---
# Incomplete — public Agent Notes import (2026-08-20)

What this change did not finish.

## Private copy still present

`ignore/Ai-notes/` still has the original dated dump. Public `.agents/notes/` is a copy with an `Origin:` header, not a git-move. Deleting the private tree is a separate `morph-scratch` commit once you are happy with the import.

The root `Ai-notes` / `ai-notes` symlinks still exist and are gitignored. Agents that ignore `AGENTS.md` may still write there.

## Classification is best-effort

Lifecycle (`proposed` vs `implemented`) and class (`architecture` vs `process` vs `testing`) were assigned from filenames and a skim, not a re-read of every campaign. Several "implemented" files are still plans. Re-file when you next touch a decision.

## Imported bodies were not rewritten

No `## Alternatives considered` was invented. Commands inside imported notes still say `Ai-notes/...` (e.g. the boundary-trajectory Python paths). The two `.py` files now live in `lab/boundary-trajectory-readout/`.

## `docs/` left alone (requested)

Pre-existing broken or stale pointers remain:

- `docs/ablation-ledger.md` and `docs/MANIFEST.md` link to `known-good-runs.md` and `data-placement-design.md` as if they lived under `docs/`. Those files were only in `Ai-notes/` and are now under `.agents/notes/implemented/`.
- Ablation ledger log columns still say `Ai-notes/`.

Fixing those is a docs edit, not done here.

## Template pieces not adopted

No `docs/constitution.md`, `lab/experiments/`, `docs/cookbook/`, or `docs/postmortem/`. No `src/` rename. No CI job for `verify_template.py`. No pytest wrapper. `CLAUDE.md` is not a symlink to `AGENTS.md`.

## Harness extras dropped

The `Ai-notes/AGENTS.md` you had open was the DeepSeek Harness spec (Chinese counterparts, TypeScript gates, INDEX ban citing notes that do not exist here). Public rules are the shorter Python-gated version in `.agents/notes/README.md`.
