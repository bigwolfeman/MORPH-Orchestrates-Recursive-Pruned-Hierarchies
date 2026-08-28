# Agent Note: Reorganization of docs/references to Per-Paper Slug Directories

Status: implemented
Class: process
Date: 2026-08-28

## Problem

The local paper reference archive under `docs/references/` previously stored loose files named by arbitrary numbers or mixed identifiers (e.g. `attention/2309.17453.md`, `attention/2309.17453.pdf`) directly inside category directories. This made browsing, reading, and maintaining companion files (.md and .pdf pairs) less clear and disorganized.

## Decision

Reorganize the entire reference tree so that every paper resides in its own slug directory under its category:
`docs/references/<category>/<paper-slug>/<paper-slug>.md` and `<paper-slug>.pdf`.

For web-only or special references like `j-space` (Anthropic), the markdown file continues directing to the online interactive publication while keeping the local PDF alongside it.

All references in `docs/references/MANIFEST.md`, `docs/references.md`, and `README.md` are updated to point to the new paths.

## Alternatives considered

- Keeping flat category directories with mixed numeric and named files: rejected because pairing companion files (.md and .pdf) and discovering papers by title/slug was cumbersome.
- Using generic filenames (`paper.md`, `paper.pdf`) inside the slug folders: rejected in favor of explicit slug names (`<slug>.md`, `<slug>.pdf`) to maintain unambiguous file names when opened in tabs or viewed in search results.

## Consequences

- Predictable, uniform file hierarchy across all 69 papers in 13 categories.
- Companion `.md` and `.pdf` files are grouped together cleanly in dedicated directories.
- All workspace links remain fully verified with zero broken references.
