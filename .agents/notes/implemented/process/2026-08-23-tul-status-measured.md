# Agent Note: TUL status — run and beats baseline

Status: implemented

## Problem

Specs and briefs still said TUL was "implemented, not yet run" / "no arm has
been TRAINED yet" after the short-schedule arms finished and A1 beat dense A0.

## Decision

Align current-state docs with `lab/tul/arms-result.md`:

- `docs/tul-spec.md` status → implemented, run, measured; further testing ongoing
- `CLAUDE.md` code map line → measured A0/A1/A3, further testing ongoing
- `docs/ablation-ledger.md` TUL section → A0/A1/A3 measured, A1r incomplete,
  remaining planned; drop "none has been RUN"
- README: drop "Disabled in config for testing" (default-off already stated)

## Alternatives considered

- Leave ledger all-planned until A1r completes — rejected; misstates the A1 win
- Rewrite arms-result into the ledger — rejected; one home for numbers stays
  `lab/tul/arms-result.md`

## Consequences

Cite arms-result for the measured comparison. Remaining ablation rows stay
planned until gated. Off-by-default in `base.yaml` unchanged.
