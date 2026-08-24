# Agent Note: MORTAR BCSR architecture readout

Status: implemented

## Problem

MORTAR BCSR is MORPH-original (no paper). Operators and README readers met it only
through the CMS lifecycle figure, CLAUDE schedule bullets, and MegaBlocks citations —
easy to confuse with CMS scoring, ReMoE, or Block-ELL.

## Decision

Ship `docs/mortar-bcsr.md` as the architecture readout for the sparse MLP path:
CMS (Taylor EMA on 16×16 tiles → 128×128 block prune under a dense mask) plus
MORTAR BCSR (carve packing, buffers, forward). ReMoE / ternary QAT stay out;
recipe integers stay in `base.yaml`. Lifecycle figure remains the schedule visual.

## Alternatives considered

- Fold into `docs/references.md` MegaBlocks entry — rejected; MORTAR is not that paper.
- Keep CMS only on the lifecycle figure — rejected; operators need the score→mask→carve
  hand-off next to the BCSR layout.
- Full paper-style draft — rejected; readout, not a paper.

## Consequences

- Cite `docs/mortar-bcsr.md` for “what is CMS + MORTAR.”
- STK lineage remains under references (MegaBlocks); this doc owns MORPH packing/wiring.
