# Agent Note: Relicense MORPH to GPL-3.0

Status: rejected — owner reverted to Apache-2.0 the next day

## Problem

The public tree shipped under Apache-2.0. A brief preference for copyleft led to
switching the root grant to GPL-3.0-only.

## Proposal

Replace the root project license with **GNU GPL v3 only** (`GPL-3.0-only`):
`LICENSE`, `pyproject.toml`, `README.md`, and `CONTRIBUTING.md` inbound terms.
Vendored `morph/sparse/stk/` (Apache-2.0) would keep its own notices.

## Alternatives considered

- **Stay / return to Apache-2.0** — permissive; chosen on revert
  ([2026-08-22-relicense-apache2](../../implemented/process/2026-08-22-relicense-apache2.md)).
- **GPL-3.0-or-later** / **AGPL-3.0** — not requested.
