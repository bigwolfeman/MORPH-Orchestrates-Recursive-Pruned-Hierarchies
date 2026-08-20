---
name: doc-standards
description: Use when writing, moving, or auditing documentation — choosing where a fact lives, or responding to verify_template.py failures.
---

# Documentation standards

## What goes where

- Standing order / how to navigate → root `AGENTS.md` and the matching section in `CLAUDE.md`
- Current behavior, specs, invariants, paper notes → `docs/` (do not reorganize that tree)
- Why we chose X, what we gave up → `.agents/notes/`
- Spikes that are not yet a decision → `lab/`
- Local scripts, wandb, Hydra outputs, bulky artifacts → `ignore/` (private, never commit)

Do not write new decision notes to `Ai-notes/` or `ai-notes/`.

## Authoring order

1. Locate the document in the tree and name its subject.
2. Keep full detail only about that subject; link children.
3. Grep distinctive phrases so the same rule is not copied. Prefer a link.

## Validate

`python scripts/verify_template.py`
