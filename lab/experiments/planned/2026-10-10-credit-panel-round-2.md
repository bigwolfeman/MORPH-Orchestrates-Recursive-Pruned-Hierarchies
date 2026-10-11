# LXTUL credit panel, round 2: ungraded fan + pointer, one-cell loop without the span decoder

Status: planned

## Question

Round 1 (`2026-10-10-lxtul-credit-panel.md`) found the one-cell loop ahead of LXTUL + pointer at
5k: CE@6 3.8997 against 3.919-3.932, write worth 0.193 against 0.154-0.168; LXTUL ahead on K1-K6
only. LXTUL's 4 cells read pairwise cos 0.95 (one lineage after the per-pass reset). Filed history
(eager, no pointer) has the ungraded fan, 4 independent lineages all written to the coda, at the
best CE of the fan family (ledger 4.334 against LXTUL 4.353 / 4.362 and the latent-selected arms
4.447-4.502). Two arms:

- **ungraded fan + pointer** (`nowta_pointer_fast`): 4 cells as 4 lineages, no z / EMA twin /
  router / teacher / reset, all 4 written, the coda's attention picks per token; everything else
  as `lxtul_pointer_fast`. Is it better than both the one-cell loop and LXTUL?
- **one-cell loop without the span decoder** (`tul_onecell_pointer_fast_nospandec`): what does the
  decoder (26 % of the step's FLOPs, 0.69 of LXTUL's loop credit) do for the simplest loop?

## Hypothesis

Distinct lineages are what the 4 cells need: the ungraded fan + pointer matches or beats the
one-cell loop on CE and write worth. The one-cell loop needs the span decoder as LXTUL did, but
loses less without it, because its single cell is also read by the coda's token CE.

## Predictions

Rulers (5k, fast recipe, seed 1, 480 rows): one-cell loop CE@6 3.8997, K1-K6 +0.0022, write
zeroed 0.193; LXTUL references CE@6 3.919-3.932 (mean 3.9245), K1-K6 +0.0088-0.0104, write zeroed
0.154-0.168. Run-to-run sd of CE@6 about 0.005 among the four LXTUL references.

| Arm | CE@6 best guess | Reading | K1-K6 best guess | write zeroed best guess |
|---|---|---|---|---|
| ungraded fan + pointer | 3.897 | beats the one-cell loop at or below 3.890; behind LXTUL at or above 3.930 | +0.003 | 0.20 |
| one-cell, no span decoder | 3.935 | the decoder carries CE for the one-cell loop if CE@6 is at or above 3.915 | +0.001 | 0.15 |

Trainer rate late: the ungraded fan within 5 % of LXTUL's (no twin forward, no router); the
one-cell loop without the decoder 15 to 25 % above the one-cell loop with it.

What each result changes:
- ungraded fan + pointer at or below 3.890: it becomes the base for the next round (Wolfe's 4
  cells, as real candidates). Between 3.890 and 3.910: a tie with the one-cell loop; the fan's
  extra cost decides. Above 3.910: distinct lineages do not pay at 5k either.
- one-cell without the decoder within 0.010 of the one-cell loop: propose dropping the decoder
  from the simple base (a large speed win). Worse by more: the decoder stays.

## Method

Configs from a frozen `git archive` of the commit that adds this file, 5000 steps, seed 1,
WANDB offline, one at a time under `gpu.lock` by `/home/wolfe/morph-scratch/abq/runner.sh`.
Readout: `core_depth_sweep.py --ckpt <name>=<config>=<step_5000> --depths 1,6 --rows 480
--batch 3`, plus the final val line (`val/plan_worth_zero`, `val/plan_worth_shuffle`,
`val/slot_cell_pairwise_cos`).
