# LXTUL credit panel, round 2: ungraded fan + pointer, one-cell loop without the span decoder

Status: failure

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

## Results

Run 2026-10-10, 5000 steps, seed 1, fast recipe, 480 rows, one run per arm. Artifacts:
`lab/experiments/results/2026-10-10-credit-panel-round-2/` (`sweep-nowta-ptr.*`,
`sweep-onecell-nospan.*`, `queue-log.txt`; the `nowta-ptr` filter also matches the round 3 second
run, so `queue-log.txt` holds lines of both).

| Arm | CE@1 | CE@6 | K1-K6 [95% CI] | write worth | cell cos | trainer tok/s |
|---|---|---|---|---|---|---|
| ungraded fan + pointer | 3.9241 | 3.9196 | +0.0045 [+0.0041, +0.0049] | 0.181 | -0.09 | 24,662 |
| one-cell, no span decoder | 3.9011 | 3.8996 | +0.0015 [+0.0012, +0.0017] | 0.186 | | 32,711 |
| one-cell loop with decoder (round 1) | 3.9019 | 3.8997 | +0.0022 | 0.193 | | 25,931 |
| LXTUL references | | 3.919-3.932 | +0.0088 to +0.0104 | 0.154-0.168 | 0.95 | 29.8k |

Predictions:
- Ungraded fan CE@6 best guess 3.897, beats the one-cell loop at or below 3.890: measured 3.9196.
  Failed. It does not beat the one-cell loop (3.8997). It also does not fall behind LXTUL (below
  3.930); it sits at the low end of the LXTUL range.
- Ungraded fan K1-K6 best guess +0.003: measured +0.0045. Held (inside the 0.006 spread).
- Ungraded fan write worth best guess 0.20: measured 0.181. Missed by 0.019. It is above LXTUL's
  0.154 to 0.168 and below the one-cell loop's 0.193.
- Ungraded fan trainer rate within 5 % of LXTUL's: 24,662 against 29.8k is 17 % lower. Failed.
- One-cell without decoder CE@6 best guess 3.935, "the decoder carries CE if at or above 3.915":
  measured 3.8996. Failed. The decoder carries no CE on the one-cell loop at 5k.
- One-cell without decoder K1-K6 best guess +0.001: measured +0.0015. Held.
- One-cell without decoder write worth best guess 0.15: measured 0.186. Failed.
- One-cell without decoder rate 15 to 25 % above the one-cell loop: 32,711 against 25,931 is 26 %
  above (21 % above the two-run mean 27,083 of rounds 1 and 3). At the edge, held against the mean.

Decision rules, applied:
- Ungraded fan above 3.910: "distinct lineages do not pay at 5k either". The fan is at 3.9196 (and
  3.9227 in its round 3 second run). The cells are distinct (cos -0.09 against 0.95).
- One-cell without decoder within 0.010 of the one-cell loop (3.8996 against 3.8997): propose
  dropping the decoder from the simple base. This held.

## Verdict

Failure. The headline prediction for the ungraded fan missed by 0.023 (3.897 predicted, 3.9196
measured) and the rate prediction missed.

Inside noise: the ungraded fan against LXTUL on CE@6 (3.9196 against the reference mean 3.9245 is
0.005). The one-cell versus fan difference (0.020) is at the single-run noise floor of 0.020 CE@6
(the winner read pair). The second fan run in round 3 (3.9227) and the one-cell second run (3.9029)
show the same 0.020 gap, so it is stable in this pair of arms; each pair differs by 0.003.

The decoder result on the one-cell loop is clear: 3.8996 against 3.8997 with a 26 % rate gain.

## Updated hypothesis

Distinct lineages do not beat the one-cell loop at 5k. They match LXTUL on CE@6 and K1-K6 without
the z machinery, so they remove cost without a CE price. The span decoder buys nothing for the
one-cell loop at 5k. Next: does the fan need the decoder? Filed in
`failures/2026-10-10-credit-panel-round-3.md`.
