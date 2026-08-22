# Agent Note: Relicense MORPH to GPL-3.0

Status: implemented

## Problem

The public tree shipped under Apache-2.0. The project owner wants copyleft so derivatives of MORPH stay open under GPL terms.

## Decision

Replace the root project license with **GNU GPL v3 only** (`GPL-3.0-only`):

- `LICENSE` is the official GPL-3.0 text
- `pyproject.toml` SPDX / classifier → `GPL-3.0-only`
- `README.md` and `CONTRIBUTING.md` inbound contribution terms match GPL-3.0

Vendored third-party trees (notably `morph/sparse/stk/`, Apache-2.0) keep their own notices. Apache-2.0 is compatible with combining under GPL-3.0; do not relicense those files.

## Alternatives considered

- **Stay Apache-2.0** — permissive; rejected by owner preference for copyleft.
- **GPL-3.0-or-later** — more flexible for future FSF revisions; rejected in favor of an explicit `GPL-3.0-only` SPDX id for a clear, frozen grant.
- **AGPL-3.0** — stronger network copyleft; not requested and heavier for research users who only train locally.

## Consequences

- Downstream distributors of modified MORPH must provide corresponding source under GPL-3.0.
- Contributors submit under GPL-3.0 (or terms that allow relicensing under it).
- Combining with proprietary-only dependencies becomes harder; Apache/MIT vendored code remains fine with attribution.
- Prior Apache-era commits are historical; current HEAD is GPL-3.0-only for the MORPH-authored tree.
