# MORPH Figure Archive

Rendered PNG previews live at the top of `docs/figures/` for quick browsing. TikZ
sources, PDFs, and LaTeX sidecar files (`.aux`, `.log`) are grouped by topic.

## Topic Index

| Topic | Diagrams |
|---|---|
| Architecture | [MORPH overview](architecture/morph_overview.tex), [MORPH block](architecture/morph_block.tex) |
| Attention | [Attention stack](attention/morph_attention.tex) |
| Embeddings | [Hybrid embeddings](embeddings/morph_embeddings.tex) |
| Memory | [Memory behavior](memory/morph_memory.tex), [Context coverage](memory/morph_context_coverage.tex) |
| Sparsity & routing | [CMS lifecycle](sparsity-routing/morph_cms_lifecycle.tex) |
| Deployment | [Deploy stack](deployment/morph_deploy_stack.tex) |

## Regeneration

Run `pdflatex` from the folder containing the target `.tex` file so LaTeX writes sidecars
next to the diagram source. Copy or regenerate PNG previews back into `docs/figures/`.
