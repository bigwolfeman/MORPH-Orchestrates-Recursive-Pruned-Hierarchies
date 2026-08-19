# MORPH Figure Archive

Rendered PNG previews live at the top of `docs/figures/` for quick browsing
(one file per diagram, no `-1` suffixes). TikZ sources, PDFs, and LaTeX sidecar
files (`.aux`, `.log`) are grouped by topic.

## Topic Index

| Topic | Diagrams | Preview |
|---|---|---|
| Architecture | [MORPH overview](architecture/morph_overview.tex), [MORPH block](architecture/morph_block.tex), [TUL overview](architecture/tul_overview.tex) | `morph_overview.png`, `morph_block.png`, `tul_overview.png` |
| Attention | [Attention stack](attention/morph_attention.tex) | `morph_attention.png` |
| Embeddings | [Hybrid embeddings](embeddings/morph_embeddings.tex) | `morph_embeddings.png` |
| Memory | [Memory behavior](memory/morph_memory.tex), [Context coverage](memory/morph_context_coverage.tex) | `morph_memory.png`, `morph_context_coverage.png` |
| Sparsity & routing | [CMS lifecycle](sparsity-routing/morph_cms_lifecycle.tex) | `morph_cms_lifecycle.png` |
| Deployment | [Deploy stack](deployment/morph_deploy_stack.tex) | `morph_deploy_stack.png` |

## Regeneration

Run from the folder that contains the target `.tex` so LaTeX writes sidecars
next to the source, then emit a single PNG into `docs/figures/`:

```bash
cd docs/figures/<topic>
pdflatex -interaction=nonstopmode -halt-on-error <name>.tex
pdftoppm -png -r 200 -singlefile <name>.pdf ../../<name>
```

Use `-singlefile` so previews are named `<name>.png`, not `<name>-1.png`.
Do not keep both an old `<name>.png` and a newer `<name>-1.png`.

When editing captions, prefer content-sized boxes (`text width` + `inner sep`)
over fixed `minimum height` panels, and place border labels with a white fill
above the border — text renames must not clip.
