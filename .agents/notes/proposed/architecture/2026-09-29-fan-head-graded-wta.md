# Agent Note: head-graded WTA for the write-all fan (arm hwta)

Status: proposed

## Problem

The write-all fan (`tul.fan_k: 4`, `tul.fan_mix: all`, arm a2,
`morph/configs/tul_slot_spandec_strict_fan4_all_fp01.yaml`) writes all four register cells
into the coda, one per prefix cell, and the coda's own per-token attention picks among
them. That read is a measured success. Without a responsibility term the cells blur into a
committee (the Thought Register read rank 1.24 of 4 with no term), so a2 carries a
winner-takes-all (MCL) term. The term is graded by the CODA: at every training step
`_tul_fan_all` runs M = 4 no-grad coda passes (cell i written alone) to find each slot's
winner by span CE, then ONE more coda pass with grad to charge the winner-alone span CE.

That grader is expensive. Measured at 5k steps: a2 ran 7783 tok/s and reached val 4.4365;
the plain slot-loop arms run 9355-10089 tok/s. The coda-graded WTA costs M no-grad coda
passes plus one grad coda pass (with its backward) per step, on top of the model's own
coda pass.

Wolfe's direction (2026-09-29): "explore everything at once and let the coda learn to pick
a winner", with a CHEAP grader.

## Proposal

`tul.fan_all_wta_grader: head` (default `coda`, bit-identical to the tree before the key).

* The grader is the parallel span head (`tul.spandec_parallel`,
  `morph/model/tul_spandec_parallel.py`): J input-free queries read one state and predict
  every token of a span at once. For every scored slot and every cell i the head reads
  cell i ALONE (`_readout(cells[:, :, i])`, the state the coda's prefix cell i is written
  from, never the mean of the cells) and gives NLL_i, its summed NLL of the SAME tokens the
  coda's WTA table scores for the slot (`fan_head_wta_targets`: the labels at the
  positions of `span_ce_index`'s bin s+1, the dump-bin slot of a row ended at max_slots
  included).
* The term is the relaxed WTA over the cells: (1-eps) on the head's argmin cell and
  eps/(M-1) on each other (eps = `fan_select_eps`), summed over the scored slots and
  divided by their token count (the coda term's normalisation), times
  `fan_all_wta_lambda`. It is folded as `fan_head_wta_weighted`, subtracted from
  train/loss and from the val loss in `train.py`. Its gradient reaches the head AND the
  cells (the loop).
* None of a2's extra coda passes run (pinned by a `_back_region` call count: 1 at train,
  the no-WTA fan's count; a2 runs 3). The coda reads all cells in its one ordinary pass and
  trains on the ordinary token CE, unchanged.
* The head's own mixture term on the MEAN of the cells is not run: the head's only
  training signal is the WTA term. The true next span is a TARGET for the head only;
  nothing built from it reaches the coda's input or the loop (pinned by perturbing the
  labels: the coda's input and the cells are bit-identical).
* The fan's refusal of the parallel head (`_check_spandec_parallel`) is lifted for this
  grader only. `_check_fan_head_grader` (called near the top of `TULConfig.__post_init__`)
  refuses it outside `fan_mix: all`, at `fan_all_wta_lambda: 0`, with `code_enum_k > 1`,
  without the head, with a detached head, with a head code table, a span cap, a target
  offset != 1, a head J below the span cap, a non-default `spandec_parallel_weight`, a
  non-default winner mode and `fan_history_streams`.

Instruments. Train: `fan/head_wta_ce` (the term), `fan/head_wta_share_k{i}`,
`fan/head_wta_winner_nll` against `fan/head_wta_mean_nll`, `fan/head_wta_dropped` (must be
0). Val (on the fan oracle's own per-cell coda table, same slots): `fan/head_coda_agree`
(fraction of slots where the head's argmin is the coda's argmin, chance 1/M),
`fan/head_pick_regret` (coda span CE of the head's pick minus the coda's best, summed and
divided ONCE by the token total, nats per token) beside `fan/rand_pick_regret` (the same
for a uniformly random cell). The oracle RAISES if the head scored a different slot set or
token count than the coda table.

Configs (5000 steps, pair with a2's 5k run):
`tul_slot_spandec_strict_fan4_all_fp01_hwta.yaml` (a2 + `spandec_parallel: true`,
`spandec_parallel_detach: false`, `fan_all_wta_grader: head`) and
`tul_slot_spandec_strict_fan4_all_fp01_nowta.yaml` (a2 with `fan_all_wta_lambda: 0`: no
WTA at all, the "coda learns to pick with no specialization pressure" control and the cost
floor).

The span-decoder decision: a2 has `tul.spandec: true` (the teacher-forced decoder grading
the mean of the cells). hwta KEEPS it; the parallel head sits beside it. Setting
`spandec: false` as the e1 chain does would remove a2's decoder target as well, a second
factor.

## Alternatives considered

* **The coda grader (a2, the default).** Correct by construction (it grades with the
  reader that is deployed) and the most expensive: M no-grad coda passes plus one grad
  pass per step.
* **The latent grader (`fan_all_wta_winner: latent`, note
  [2026-09-29-latent-wta.md](2026-09-29-latent-wta.md)).** An InfoNCE score in latent space
  against the true next span's pooled prelude states; no coda pick pass, but the coda grad
  pass stays. The offline probe measured its agreement with the coda's winner at chance
  (0.19-0.28 against 0.25). The head grader differs in two ways: it scores the cells on the
  coda table's own TOKENS (a likelihood, not a latent similarity), and it drops the coda
  grad pass as well, because the WTA gradient flows through the head.
* **Map-winner sharing (`fan_all_wta_winner: map`).** Halves the coda row count under code
  rollouts; needs `code_enum_k >= 2` and still runs coda pick and grad passes.
* **The head's hard WTA with a random eps winner (a2's eps rule).** Rejected for the
  relaxed form: all M NLLs exist with grad at no extra cost, so the relaxation removes the
  eps draw's variance. The weights differ in the third decimal (0.95 vs a2's expected
  0.9625 on the argmin at eps 0.05, M 4); named in the config header.
* **Replacing the teacher-forced span decoder with the parallel head (`spandec: false`).**
  Rejected for this arm: two factors at once (see Proposal).

## Acceptance criteria

* CPU contracts pass: `tests/test_tul_fan_head_wta.py` (default is the tree, one coda pass,
  cell-alone reads, the weights, the gradient path, the target never reaching the coda,
  the dump-bin target, real dropout, every refusal, the configs, the val-loss subtraction
  and a hand recomputation of the val instruments).
* The 5k run of hwta, paired with a2's 5k run (same seed and data): val CE within a2's
  seed noise of 4.4365 at a tok/s clearly above a2's 7783, measured on the same machine.
  The nowta arm gives the cost floor and the CE the WTA pressure buys; if hwta sits at
  nowta's CE, the head grader bought nothing.
* `fan/head_coda_agree` clearly above 1/M and `fan/head_pick_regret` clearly below
  `fan/rand_pick_regret` at val. A head at the random floor is not grading.
* `fan/oracle_ce` below `fan/single_ce` by at least a2's gap: the cells still specialize.

## Risks

* **The head can be gamed.** The cells can learn to be easy for a 2-block committed reader
  instead of useful to the coda. The instrument is `fan/head_coda_agree` falling toward
  1/M and `fan/head_pick_regret` rising to `fan/rand_pick_regret`.
* **A committed product reader sees less than the coda.** It is capped at the product of
  the span's marginals (`product_reader_le`), and at Stage 1 its CE at span offsets >= 4
  sat near the unigram (`lab/experiments/failures/2026-09-24-lxtul-e-stage1.md`). The
  cell it prefers may not be the cell the coda uses.
* **Train/deploy.** None for the READ: at deploy there is no grader; the coda reads all
  cells exactly as at train. The head is training-only and absent from generation.
* **The head is NOT cheap at this shape (first measurement, 2026-09-29).** 41-step smokes
  on the 5090, run one after another on the same machine, during warmup (lr 4e-6 at step
  40): hwta 7093 tok/s, a2 6876, nowta 9503 at step 40 (peak 14.19 / 14.84 / 13.77 GB).
  So the head grader removes only about 8 % of the WTA's cost over the no-WTA floor. The
  head reads M x (~50 slots x B) rows at J = 32 through 2 blocks and runs a vocabulary GEMM
  over M x the scored tokens (~24k rows x 49k vocab), checkpointed (forward twice). That
  GEMM is the likely cost; it is not profiled. Levers before a 5k run: a smaller head, a
  sampled vocabulary, or a capped target (refused today, because the grade must cover the
  coda table's tokens). Steps 20-40 are too early to stand for a run's average.
* **RNG.** hwta draws neither a2's extra dropout masks nor its eps draw, so the two arms
  are not RNG-aligned after step 0; the pairing is by seed and data, not by stream.
