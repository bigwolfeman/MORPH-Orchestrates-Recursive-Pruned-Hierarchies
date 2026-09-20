# Planned: the fan with select-then-commit — every stream gets a reader

Status: planned

Arm: `slot-spandec-strict-fan4-select` (`morph/configs/tul_slot_spandec_strict_fan4_select.yaml`),
ONE factor over `slot-spandec-strict-fan4-epivol`: `tul.fan_mix` softmax → select
(`fan_select_eps 0.05`, `fan_select_gate_lambda 1.0`). Partners: `fan4-epivol` (the
mixture arm, `../failures/2026-09-20-lxtul-fan4-epivol.md`) and the prefix_k 4 width
partner `slot-spandec-strict-pk4` (`../failures/2026-09-19-lxtul-fan4.md`).
Note: `.agents/notes/proposed/architecture/2026-09-20-fan-select-then-commit.md`.

## Question

`fan4-epivol` produced four live, input-dependent, near-orthogonal streams per slot
(rank 2.82 of 4, `axis_cos` 0.14, norms 12.3 / 10.4 / 9.3 / 11.5) and the coda still read
one of them: the softmax mixture concentrated on stream 0 (entropy 0.41, `oracle_pick0`
0.71), stream 0 alone sat within 0.005 of the mixture and streams 1 to 3 read 0.14 to
0.42 nats worse. The coda was trained on the mixture only, so that reading is a reader
that never saw the other three streams, and the oracle (0.018 over the mixture) is a
lower bound taken with the wrong reader (`the-reader-was-the-limit`). **When every stream
has been written alone into the coda during training, does an oracle over the K streams
beat the best single stream, and does a learned selector capture that headroom?**

## Hypothesis

A shared reader plus a soft gate is a rich-get-richer loop: the stream the coda reads
best gets more weight, more gradient, and becomes more readable. Writing ONE chosen stream
alone per slot, with the choice made by the coda's own span CE and an epsilon of random
writes, gives every stream a reader. If the streams' diversity carries anything the coda
can use, the per-slot oracle will then beat the best single stream by more than the
noise floor, and the gate, trained to predict the winner, will recover part of it at
inference. If, with every stream readable, the oracle still sits on the best single
stream, the streams carry nothing selectable and that is the honest closing.

## Predictions (frozen)

At the FINAL val (`*_final` keys at step 5000; the periodic val at 4750 is the fallback),
scored by `../results/2026-09-19-lxtul-fan4/fan_score.py` and the stream probe. Under
`select`, `fan/mixed_ce` is the SHIPPED write, the gate's argmax stream written alone.

- **P-1 (every stream is readable).** Every `fan/stream_ce_k{i}` within **0.05** nats of
  the best of the four (epivol: 0.005 / 0.138 / 0.423 / 0.231 above `mixed_ce`). **60 %.**
- **P-2 (THE FALSIFIER: the oracle has headroom over the best single stream).**
  `fan/oracle_ce` below `min_i fan/stream_ce_k{i}` by more than **0.022** nats. **40 %.**
  If it fails while P-1 holds, the K streams are four readable copies: nothing to select
  over at the span level on this corpus, with every stream given a reader.
- **P-3 (the gate captures headroom).** `fan/mixed_ce` (the gate's argmax written) below
  `min_i fan/stream_ce_k{i}` by more than **0.005** nats, and `fan/gate_agree` > **0.40**
  (1/K = 0.25 is a gate that knows nothing). **35 %.**
- **P-4 (the shape holds).** `fan/stream_rank_t1` > **2.0** of 4; the stream probe on the
  step-5000 checkpoint reads `axis_cos` < **0.5** at pass 1. **65 %.**
- **P-5 (against the width partner).** Paired depth-6 token CE on the same 480 rows:
  `select − pk4` < **+0.005** (epivol read +0.0073 worse against pk4 at the fan4 filing's
  method; the ruler's seed spread is 0.024). **35 %.**
- **P-6 (the winner is not one stream).** Train-side `fan/select_pick0` between **0.15
  and 0.60** over the last 1000 steps, and no `fan/select_share_k{i}` below **0.05**. **55 %.**
- **P-7 (depth).** Token K1−K6 > **+0.005**; K3−K6 > **+0.001**. **25 %.**
- **P-8 (cost and stability).** Step rate at step 200 ≥ **6,000** tok/s (epivol 10,415;
  the K no-grad coda passes are the cost); no tripwire; `fan/select_gate_weighted` below
  **10 %** of `loss/total` after step 1000. **70 %.**

**NOT predicted:** the arm's absolute CE, other K, other eps or gate weight, transfer to
math or Sudoku, the behaviour of a select arm without the epivol term.

## Method

Trained on the 5090 through `run_recon.sh` (seq 1024, batch 6, 5000 steps, sweeps
2500 and 5000, kind `slot`, outdir `2026-09-19-lxtul-fan4`), the same recipe as every
fan arm. The runner's readouts run the bare config, which is this arm's config (no
override), so its sweeps, worth profile and state probe are the runner's own. Readings in
order: (1) `fan/stream_ce_k{i}` against each other (P-1); (2) `fan/oracle_ce` against the
best stream (P-2) and `fan/mixed_ce` with `fan/gate_agree` (P-3); (3) the 3070 stream
probe (`lab/divergence/fan_stream_probe.py`, 48 rows, depth 6) and `fan/stream_rank_t1`
(P-4); (4) the paired depth-6 gap against pk4 from the two sweeps' `tokens.npz` (P-5, the
fan4 filing's block bootstrap); (5) the train-side `fan/select_*` series (P-6, P-8); (6)
the K-curve (P-7). Instruments: `tests/test_tul_fan_select.py` (12 passed, CPU) proves
the hard gather, the argmin, the eps guard, the gate loss, and that a zero gate's eval
write equals stream 0's own replay exactly.

## Not verified before launch

- The GPU smoke of the composed config (recorded below before the queue line).
- The step-time cost of the K no-grad coda passes at seq 1024, batch 6 (P-8 reads it).
- Whether `fan_select_eps 0.05` is enough to keep a loser readable; P-1 and P-6 read it.
