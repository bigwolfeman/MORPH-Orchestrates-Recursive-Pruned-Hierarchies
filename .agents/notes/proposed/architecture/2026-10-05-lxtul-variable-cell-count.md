# Agent Note: LXTUL variable cell count, Poisson-sampled at train, free at inference

Status: proposed

## Problem

Wolfe (2026-10-05), verbatim: "For lxtul cell count, we should try applying poisson
sampling to the number of cells to see if it generalizes. This whole thing needs to be
tested for its behaviors. For instance can I train at 4 and inference at 128 fine? a
variable cell count at inference would be huge."

What cell count is in LXTUL today (checked in the code 2026-10-05): under the
latent-selected loop the cells are a SEARCH, not a channel. Every pass, the slot's M cells
are reset to the router's winner and the next pass explores M variations around it; at
the exit the coda reads the final winner alone, and the losers' positions are exactly zero
(`_fan_route_cells`, `transformer.py` ~13627). So M is the search breadth per pass. A cell
count that can change at inference is a test-time compute knob: more proposals per pass
for a hard span, fewer for an easy one, with no retraining. The loop's depth already works
this way (Poisson depth at train, any depth at eval).

## Proposal

1. **Train with a sampled cell count.** M ~ Poisson(mean 4) clamped to [2, M_max] (per
   batch first; per slot if the shapes allow), the way `mean_depth` / `max_depth` sample
   the loop's depth.
2. **Measure generalisation at inference.** Train at a fixed 4 and at Poisson(4); evaluate
   both at M in {1, 2, 4, 8, 16, 32, 64, 128}. Readings per M: K1-K6, ledger CE, the gap to
   plain, the router's pick versus the oracle over cells, tok/s. The question: does CE fall
   (or at least hold) as M grows past the training value, and does the Poisson arm extend
   further than the fixed-4 arm?
3. **Behaviour checks** (Wolfe: "This whole thing needs to be tested for its behaviors"):
   router calibration at large M (does the pick degrade toward random as candidates grow?),
   cell diversity versus M (does the epivol term's effect hold at M it never saw?), memory
   and speed scaling, and whether the 8-cell result (`lxtul_fan8`) already shows the trend.

Design blockers to solve first. Today several parts are tied to a cell INDEX, so M cannot
change without new code:

- the register's seeds: one learned query `Q_i` and one embedding `P_cell[i]` per cell
  (`TULSlotRegister`), and each cell's own per-pass injection seed `e_i`;
- the router's load-balance `bias` `[M]` buffer (`FanRouter`); its score itself is shared
  across cells and does not depend on M;
- `W_prefix[i]`, one projection per cell index (the winner is written through its index);
- the epivol diversity term and the instruments, sized at build.

An index-free seed is needed: for example one shared query plus a sampled perturbation per
cell, or a seed drawn from a learned distribution. The write would go through one shared
projection.

## Alternatives considered

- **Fixed cell count, tuned once** (4 today, 8 in `lxtul_fan8`). Simpler; gives no
  inference-time knob.
- **Learned per-slot cell count** (a halting-style policy). Not proposed: Wolfe's standing
  rules forbid depth curricula and learned-sigma arms, and a sampled count is the analogue
  of the sampled depth that already works.

## Acceptance criteria

- An index-free cell seeding, router and write, with M a per-forward argument, and a test
  that M = 4 under the new code reproduces the fixed-4 model's forward when the seeds match.
- The two training arms (fixed 4, Poisson 4) each have a prereg, and the M sweep is filed.
- CLAUDE.md keeps a reminder until this note leaves `proposed/`.

## Risks

- Index-free seeds may lose what the distinct per-cell queries bought (the 2026-09-13
  register: distinct seeds were the point).
- Large M multiplies loop cost linearly; M = 128 at inference is 32x the loop's work at 4.
- The router may not rank 128 candidates it was never trained against.
