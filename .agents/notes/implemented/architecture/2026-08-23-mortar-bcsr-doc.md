# Agent Note: MORTAR BCSR architecture readout

Status: implemented

## Problem

MORTAR BCSR is MORPH-original (no paper). Operators and README readers met it only
through the CMS lifecycle figure, CLAUDE schedule bullets, and MegaBlocks citations —
easy to confuse with CMS scoring, ReMoE, or Block-ELL.

## Decision

Ship `docs/mortar-bcsr.md` as the single architecture readout for the format:
pre/post carve storage, 128×128 geometry, buffer names, carve/forward pseudocode,
scope, explicit “not CMS / not ReMoE / not Block-ELL,” and operator gotchas.
Link from `docs/MANIFEST.md`. Recipe integers stay in `base.yaml`. CMS schedule
story stays on the lifecycle figure / pruning module.

## Alternatives considered

- Fold into `docs/references.md` MegaBlocks entry — rejected; MORTAR is not that paper.
- Expand the CMS lifecycle figure caption into a full writeup — rejected; figure is
  schedule, not format semantics.
- Full paper-style draft — rejected; user asked for a readout, not a paper.

## Consequences

- Cite `docs/mortar-bcsr.md` for “what is MORTAR.”
- STK lineage remains under references (MegaBlocks); this doc owns MORPH packing/wiring.
