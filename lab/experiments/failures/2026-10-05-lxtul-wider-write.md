# Planned: does writing each LXTUL cell into 2 coda positions narrow the gap to plain?

Status: failure

Date: 2026-10-05 01:20 CDT, before the smoke and the run. Config `lxtul_fan4x2`: `lxtul.yaml` with
`tul.prefix_per_cell: 2` and `tul.prefix_k` 4 -> 8, 5000 steps; the composed config differs
from lxtul only in those two keys, `training.steps` and `wandb.name`
(`tests/test_prefix_per_cell.py`). Wolfe (2026-10-05): width first, "make sure this goes to
the front of the queue". Item 2 of the
[CE queue](../../../.agents/notes/proposed/architecture/2026-10-04-lxtul-ce-test-queue.md).

## Question

Under LXTUL's latent-selected loop the coda reads ONE live vector per earlier span: the
final winner, through `W_prefix[winner]`; the losers' positions are zero. The exact-recall
probe put 65 % of the 10k gap on bigram repeats 33-256 tokens back. The 8-cell arm
(more search, still one live position) left the gap unchanged (+0.270) and halved K1-K6
(+0.0103). This arm keeps the loop at 4 cells and writes the winner into 2 positions
through 2 projections (copy 0 identity, copy 1 a random orthogonal start, both trainable).
Does a second readable position per span let the coda pull more out of the same cell?

## Predictions (mine, orchestrator)

References: LXTUL 5k seed 1 / seed 2: gap +0.272 / +0.282, worth +0.176 / +0.154, K1-K6
+0.0215 / +0.0172, 9800 / 8876 tok/s (ckpt every pass). The 10k run at this arm's
`ckpt_grad_iters: 4` ran 12402 tok/s.

- **W-1**: no detonation (HEALTHY or one recovered spike). 80 %.
- **W-2**: gap to plain 5k below +0.262. 40 %.
- **W-3**: K1-K6 > +0.0108, CI above 0 (the loop is unchanged). 70 %.
- **W-4**: zero-cell worth above +0.190. 40 %.
- **W-5**: tok/s at least 11000 (within about 10 % of the 10k run's 12402 at the same
  checkpointing). 65 %.
- **W-6**: mean-ablating the cells' shared direction changes K6-K1 by under 30 %. 65 %.

Pass rule: W-2 and W-3 hold.

My expectation: small. The span enters each cell through one single-head attention pool
(a weighted average of the span's prelude states, `TULSlotRegister`), so a second view of
the same cell cannot hold tokens the pool blurred. If W-2 fails, the encode side (queue
items 9 and 10) is the next test, not wider read-outs (item 11).

## Method

5000 steps, seed 1, `runner_steps.sh`, at `model.ckpt_grad_iters: 4` (lxtul's). A 60-step
smoke with a val pass first, for memory; if it runs out of memory, checkpoint every pass
and amend this Method before the run. Readouts as for every LXTUL arm: depth sweep (480
rows), gap vs plain 5k, worth, the exploration ledger, the LayerNorm probe (geometry and
`--vablate`), the repetition eval, and the exact-recall probe against plain 5k.

## Results

References as preregistered: LXTUL 5k seed 1 / seed 2 K1-K6 +0.0215 / +0.0172, gap to plain 5k +0.272 / +0.282, zero-cell worth +0.176 / +0.154, far-bigram bucket gap +1.150 / +1.154. Readouts: 480-row depth sweep, `paired_vs_ruler.py` gap vs plain 5k, `worth_profile.py` on 192 rows, the exploration ledger, `exact_recall_gap.py` vs plain 5k, the LayerNorm probe's `--vablate` on 96 rows (K6-K1 change when the cells' shared-direction amplitude is set to its mean). Artifacts: [`../results/2026-10-05-lxtul-wider-write/`](../results/2026-10-05-lxtul-wider-write/).

| reading | wider write (each cell into 2 coda positions) |
| --- | --- |
| tripwire | AMBIGUOUS, max 2.45e3 at 1035 (below the 1e4 rule) |
| K1-K6 | +0.0171 [+0.0162, +0.0181] |
| gap to plain 5k | +0.2835 [+0.2678, +0.3011] |
| zero-cell worth / shuffle | +0.158 / +0.149 |
| ledger CE | 4.3603 |
| far-bigram bucket gap | +1.172 |
| mean-ablation K6-K1 change | -5.6 % |
| tok/s, peak | 11156, 22.9 GB |

| prediction | held |
| --- | --- |
| W-1 no detonation | yes |
| W-2 gap < +0.262 | no |
| W-3 K1-K6 > +0.0108 | yes |
| W-4 worth > +0.190 | no |
| W-5 tok/s >= 11000 | yes |
| W-6 mean-ablation change < 30 % | yes |

## Verdict

Failure: W-2 fails. A second, rotated view of the same winner cell adds no CE; the far-bigram bucket
is slightly worse. This matches the Lean result (`exact_tuple_write_needs_dim`): a linear write of
one cell cannot deliver more exact tokens than the cell holds. Read-out width is not the limit.

## Updated hypothesis

As preregistered: the limit is upstream of the write. Item 11 (m = 4 / 8) is struck; the encode-side
tests ran as the cell-packing panel.
