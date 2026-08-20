# Agent Note: Private Scratch Repo Setup
Status: rejected — historical scratch imported from private Incomplete
Origin: Ai-notes/Incomplete/2026-07-03-private-scratch-repo.md
Imported: 2026-08-20 from private Incomplete import pass

---
# Incomplete — private scratch repo setup (2026-07-03)

- Public MORPH changes not committed: `.gitignore` (symlink patterns) and `CLAUDE.md` (scratch-repo blurb). Commit on the public side when ready.
- `ignore/` is owned by `root` on this volume, so git reports "dubious ownership". Commands need `GIT_CONFIG_KEY_0=safe.directory` / `GIT_CONFIG_VALUE_0=<ignore path>` (or a local `safe.directory` entry). Not set globally per policy.
- No CONTRIBUTING.md blurb — only CLAUDE.md was updated.
- Regenerable megatraces (`*.m2g`, `*.optlog`, `perf/*.json`, `fullmodel_*_trace.json`) were deleted, not archived.
