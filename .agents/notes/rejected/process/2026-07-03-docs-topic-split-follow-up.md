# Agent Note: Docs Topic Split Follow-Up
Status: rejected — historical scratch imported from private Incomplete
Origin: Ai-notes/Incomplete/docs-topic-split-2026-07-03.md
Imported: 2026-08-20 from private Incomplete import pass

---
# Docs Topic Split Follow-Up

## What Was Not Done Properly Initially

The first cleanup pass moved markdown references and TikZ source/sidecar files, but missed
root-level reference PDFs and rendered figure PNGs because the normal file search did not
surface those generated or ignored assets.

## Current State

The missed PDFs have now been moved into the same topic folders as their markdown copies.
Rendered figure PNG previews were restored to the top of `docs/figures/` for quick
browsing, while the figure `.tex`, `.pdf`, `.aux`, and `.log` files remain topic-nested.

`docs/references.md` was also missing archive papers that already lived under
`docs/references/`: Token Superposition Training (2605.06546, active), Semantic Step
Prediction (2604.18464, removed STP lineage), and a prose entry for Nested Learning
(2512.24695, table-only before). GLA (2312.06635) was active in code but absent from both
the paper map and the archive; PDF was downloaded and a short MD stub written (full-text
markdown extraction tools were not available in this environment).

Figure cleanup: removed outdated Jun-13 non-`-1` PNG duplicates and the Jul-1
`*-1.png` pdftoppm artifacts; each diagram now has one top-level `<name>.png`.
Clipping from caption renames (attention CCA prologue sitting on the strip border,
fixed-height overview side panels) was fixed in the TikZ sources and re-rendered.
Some boxes still run close to their borders at 200 dpi; further padding is optional.
