# Agent Note: Incomplete Import Policy

Status: implemented

## Problem

Historical markdown in private `Incomplete` contains mixed-quality process scratch that is useful for provenance, but not all of it represents durable accepted decisions. Without a policy, imports can become a flat dump and blur lifecycle meaning in `.agents/notes/`.

## Decision

Treat imports from private `Incomplete` as historical rejected process notes by default and place them under `.agents/notes/rejected/process/` with explicit `Origin` and `Imported` metadata. Keep new operational leftovers in `Incomplete` unless they document a durable decision, rationale, or invariant worth promoting into `.agents/notes/`.

## Alternatives considered

1. Import `Incomplete` content directly into implemented notes.
   - Rejected because most entries are scratch follow-ups, not accepted durable decisions.
2. Keep all `Incomplete` content private only and never import.
   - Rejected because provenance and historical context are sometimes needed in the public record.
3. Import everything into a flat folder.
   - Rejected because it weakens lifecycle/class organization and makes retrieval harder.

## Consequences

Imported historical leftovers remain searchable and traceable while clearly marked as rejected process scratch. `.agents/notes/` stays structured by lifecycle and class, and future unfinished operator leftovers remain in `Incomplete` unless promoted by explicit durable-decision criteria.
