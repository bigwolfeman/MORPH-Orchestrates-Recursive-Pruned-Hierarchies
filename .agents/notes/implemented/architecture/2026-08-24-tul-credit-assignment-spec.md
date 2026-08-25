# Agent Note: TUL credit-assignment pseudocode in spec

Status: implemented

## Problem

The forward layout was in `docs/tul-spec.md`, but how CE on tokens reaches the
weight-shared core (route A vs summed coda attention route B, then truncated
BPTT through `J_core`) lived only in chat.

## Decision

Add `docs/tul-spec.md` §5.1 with the forward/backward credit-assignment
pseudocode. Point `lab/tul/README.md` at it. Deduped the pasted double
coda/`lm_head` block; softened hard-coded 4/6/`1024` to config-relative wording.

## Alternatives considered

- New file under `lab/tul/` — rejected; one home for the contract is the spec.
- Put under §3 only — rejected; this is about loss routes, not layout alone.

## Consequences

Cite §5.1 for “how does the backbone get gradients under TUL.”
