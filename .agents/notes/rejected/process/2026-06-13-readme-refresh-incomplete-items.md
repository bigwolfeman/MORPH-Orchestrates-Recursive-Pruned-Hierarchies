# Agent Note: README Refresh Incomplete Items
Status: rejected — historical scratch imported from private Incomplete
Origin: Ai-notes/Incomplete/2026-06-13-readme-refresh.md
Imported: 2026-08-20 from private Incomplete import pass

---
# README Refresh Incomplete Items

## What Was Not Fully Validated

- I did not run a training smoke test. The README update is documentation-only and was verified against configs, source files, and linter diagnostics.
- `docs/references.md` still contains older architecture wording in its introduction. I updated README references to be high-level, but did not rewrite the references document in this pass.

## Verification Completed

- Checked `README.md` with IDE lints: no linter errors reported.
- Confirmed the rendered PNGs referenced by `README.md` exist in `docs/figures/`.
- Cross-checked the README content against `morph/configs/base.yaml`, curriculum configs, `morph/model/transformer.py`, `morph/model/sparsity.py`, `morph/model/routing.py`, `morph/training/pruning.py`, and figure sources.
