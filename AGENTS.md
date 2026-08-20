# AGENTS.md

Standing orders. Detail lives in the linked file. `CLAUDE.md` holds the model/runtime brief; do not copy it here.

## Documentation map

| Kind of work | Where it goes |
|---|---|
| Model/runtime gotchas, recipe pointer | [CLAUDE.md](CLAUDE.md) — recipe numbers live in `morph/configs/base.yaml` |
| Specs, invariants, paper notes, figures | `docs/` — leave that tree's layout alone; link into it |
| Decision records (why, what we gave up) | [.agents/notes/](.agents/notes/README.md) |
| Production code | `morph/` |
| Tests | `tests/` |
| Spikes | `lab/` |
| Local artifacts, wandb, Hydra outputs | `ignore/` (private git repo, never commit) |
| Operator leftovers | `Incomplete/` (gitignored) |

**Do not write notes to `Ai-notes/` or `ai-notes/`.** Those paths are gitignored leftovers. New notes: `.agents/notes/{lifecycle}/{class}/yyyy-mm-dd-slug.md`. Read [.agents/notes/AGENTS.md](.agents/notes/AGENTS.md) once per context window when writing notes.

## Standing orders

- **One home per fact.** Put a rule in the tier that owns it; elsewhere, link.
- **Non-trivial changes get an Agent Note in the same change.** Exempt only mechanical or local edits.
- **Ai-notes migration lookup lives here:** [.agents/notes/implemented/process/2026-08-20-ai-notes-migration-manifest.md](.agents/notes/implemented/process/2026-08-20-ai-notes-migration-manifest.md).
- **Misconfiguration and failed runs fail loud.** Do not swallow errors or skip missing referents.
- **Document current state** in `docs/` and `CLAUDE.md`. Change stories belong in commits or Agent Notes.
- After adding or moving notes, run `python scripts/verify_template.py` and report the result.

## Editing these instructions

Keep each rule short. Subtree `AGENTS.md` files hold only rules that do not belong at root. `morph/model/CLAUDE.md` is the hot-path subtree brief.
