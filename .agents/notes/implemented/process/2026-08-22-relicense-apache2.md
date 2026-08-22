# Agent Note: Restore Apache-2.0 as the project license

Status: implemented

## Problem

MORPH briefly switched to GPL-3.0-only. The owner reversed that preference; the
public tree should be Apache-2.0 again.

## Decision

Restore **Apache License 2.0** as the root project license:

- `LICENSE` is the official Apache-2.0 text
- `pyproject.toml` SPDX / classifier → `Apache-2.0`
- `README.md` and `CONTRIBUTING.md` inbound contribution terms match Apache-2.0

Vendored third-party trees (notably `morph/sparse/stk/`) keep their own notices —
unchanged, also Apache-2.0.

Supersedes
[rejected/process/2026-08-21-relicense-gpl3](../../rejected/process/2026-08-21-relicense-gpl3.md).

## Alternatives considered

- **Keep GPL-3.0-only** — copyleft; rejected by owner preference.
- **GPL-3.0-or-later / AGPL-3.0** — not requested; would keep copyleft.

## Consequences

- Downstream may use MORPH under Apache-2.0 terms again (including proprietary
  derivatives, with attribution).
- Contributors submit under Apache-2.0 (or Apache-2.0-compatible terms).
- Any remote that still shows GPL-3.0 needs a refresh after this lands on HEAD.
