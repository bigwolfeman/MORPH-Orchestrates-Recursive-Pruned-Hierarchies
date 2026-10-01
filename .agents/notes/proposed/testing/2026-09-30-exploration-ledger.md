# Agent Note: the exploration ledger (price a fan's selection and search inside one checkpoint)

Status: proposed

## Problem

The fan arms (the ungraded fan, the coda-graded fan, the factor fan, both routers, the
latent-selected loop) are compared to each other by val CE at 5000 steps, one seed each.
Two runs of one arm differ by about as much as two arms do
(measured: runs at a fixed seed decorrelate within 11 steps), and the gaps
between fan arms are 0.01 nats. So an arm-vs-arm CE gap cannot say what a fan's selection
or its search across passes contributes. Wolfe, 2026-09-30: "How can we better prove
latent exploration contribution and effect in these short training runs?"

## Proposal

`lab/divergence/exploration_ledger.py`: one checkpoint, the same 480 validation rows
`core_depth_sweep.py` reads, one intervention at a time, each per-token CE paired with the
model's own forward on the same tokens, a 2000-draw row bootstrap on every delta and a
split by the plain model's per-span CE quartile (exploration should pay on hard spans; an
ensemble pays everywhere).

Readings, where the arm has the mechanism:

| reading | what it removes | anchor |
| --- | --- | --- |
| random_exit | the learned pick (random cell read) | 0 = pick worth nothing |
| random_search | the latent-selected loop's search (random winner every pass) | 0 = search worth nothing |
| teacher | the router's error (follow the latent teacher) | < 0 = router headroom |
| cell_i, single_cell | the pick or the width (cell i alone, mean over i) | > 0 = pick or width value |
| oracle | nothing: the per-span best cell | selecting arms <= 0 (headroom); write-all fans > 0 = all cells beat the best one |
| no_reset | the reset to the winner (M independent chains, pick at exit) | > 0 = searching from the winner pays |
| fixed_lineage | all selection (cell 0 at every pass) | > 0 = selection value |
| loser_t{t} | the pick at pass t only (a random non-winner) | > 0 = credit of pass t |

No `morph/model` change: every intervention is an instance-level patch installed and
removed by `ledger_patch` (the latent-selected loop's `tul_fan_lsel_router.select`, the
module-level `reset_to_winner`, a router's `_tul_fan_route` winner, a write-all fan's
`prefix_project(cells=...)`), so the probe runs on any checkpoint at any commit that has
the arm. The CE comes from the labelled eval forward (the target twin feeds the teacher),
captured at `_tul_group_losses` and turned into logits the way the label-free path does.
Queue: `/home/wolfe/morph-scratch/abc/after_lsel_ledger.sh`, after the latent-selected loop
arms, on eight fan arms.

## Alternatives considered

- **Seed replicates of every arm.** Prices the arm, not the mechanism, at 3x the GPU time
  per arm, and still cannot separate selection from width from search.
- **Add eval-only switches to the model (`lsel_follow="random"` etc.).** Puts probe code on
  the shipped path and ties the probe to one commit. The patch object does the same with
  no model change; tests pin that the identity patch is the shipped forward bit for bit.
- **Span-level CE from the val oracle table only.** The oracle already sums span CE per
  forced cell, but it has no per-token output, so no row bootstrap and no quartile split,
  and it covers only the forced-cell readings.
- **Score the interventions by cosine or latent R^2.** Refused (Wolfe 2026-09-22: never pass or fail on a
  cosine); latent readings stay diagnostics, the ledger reads token CE.

## Acceptance criteria

- `tests/test_exploration_ledger.py` passes on one CPU core; a broken override fails it
  (sabotage run 2026-09-30: shifting the loser pass and removing the no-reset patch each
  fail one test).
- Self-checks in every JSON: captured CE equals the forward's `ce_tokens` per batch
  (raise above 1e-2), and the label-free forward's CE on the first batch.
- The GPU smoke runs on a router, a write-all fan and the latent-selected loop.

## Risks

- `cell_i` forces cell i in EVERY slot of the row at once (the val oracle's convention),
  so a token of span j reads forced cells from every earlier slot. The oracle is therefore
  a per-span minimum over four whole-row forwards, not a per-slot search.
- The patches depend on private names (`_lsel_pass` calling `select` once per pass,
  `_tul_group_losses` called once per forward). A refactor that changes either breaks the
  ledger loudly: the pass count and the capture count are asserted per forward.
- `loser_t{t}` at T = 6 costs five extra forwards; the latent-selected loop's ledger is
  about 15 forwards of 480 rows.
- The plain join uses `detect_shift`; an ambiguous alignment drops the quartile split and
  says so in the JSON rather than guessing.
