# Agent Note: Move TUL campaign clutter out of docs/

Status: implemented

## Problem

`docs/` root mixed the canonical TUL contract with arm results, RCAs, spike evals, and
follow-on arm specs. That crowded the human docs navigator and blurred "current
contract" vs "campaign evidence".

## Decision

Keep only [`docs/tul-spec.md`](../../../../docs/tul-spec.md) in `docs/`. Place the rest by
kind:

| Kind | Destination |
|---|---|
| Arms comparison, divergence RCA, trace-inverter spike | [`lab/tul/`](../../../../lab/tul/) |
| Arm CW contract (shipped eval + `coda_token_cut`) | [archived/architecture/2026-08-18-tul-compaction-window.md](../architecture/2026-08-18-tul-compaction-window.md) |
| Arm D teacher-distill (unbuilt) | [proposed/architecture/2026-08-18-tul-teacher-distill.md](../../proposed/architecture/2026-08-18-tul-teacher-distill.md) |

Update inbound links in README, CLAUDE, configs, model comments, and
[`docs/MANIFEST.md`](../../../../docs/MANIFEST.md).

## Alternatives considered

- **`docs/tul/` subdirectory** — still restructures the docs tree; standing order is to
  leave that layout alone and link out.
- **Everything into `lab/`** — loses Agent Note lifecycle for CW/D decisions that still
  guide code and proposed work.
- **Everything into `.agents/notes/`** — notes README forbids run logs and postmortems;
  arms-result and divergence-rca are exactly those.

## Consequences

- `docs/` root is the stable surface; TUL evidence is one `lab/tul/` hop away.
- Code comments that cited `docs/tul-compaction-window-spec.md` now cite the Agent Note
  path; `docs/tul-spec.md` citations are unchanged.
