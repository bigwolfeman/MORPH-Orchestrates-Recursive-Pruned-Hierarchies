# Planned: the fan with select-then-commit — every stream gets a reader

Status: failure

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

- The GPU smoke of the composed config: Spark, 2026-09-20 18:29 to 18:38 UTC, the runner's
  12-step command at ec0aa41, exit 0, peak 17.63 GB (epivol's smoke 17.58), build banner
  `mix='select' repel_mode='epivol'`, final val `fan/gate_agree` 0.4804, `fan/mixed_ce`
  11.1539, `fan/oracle_ce` 11.1373 (355 spans). The GPU shared the card with a running
  readout, so its step rate is not a reading. The train-side `fan/select_*` keys are
  proven present in the forward's output by `tests/test_tul_fan_select.py`; their wandb
  rows could not be read from the offline smoke and are checked on the 5090 run's first
  logged steps.
- The step-time cost of the K no-grad coda passes at seq 1024, batch 6 (P-8 reads it).
- Whether `fan_select_eps 0.05` is enough to keep a loser readable; P-1 and P-6 read it.

## Results

Run: 5,000 steps at `52457da` on the 5090 through `run_recon.sh`, started 14:17 local
2026-09-20, DONE 15:52, tripwire HEALTHY (max 92.3 at step 219), final val_loss 4.5582,
RATE 8,018 tok/s at step 200. Artifacts in `../results/2026-09-19-lxtul-fan4/`:
`sweep_slot-spandec-strict-fan4-select_{2500,5000}.json`, `worth_…_5000.json`,
`slot_state_…_5000.json`, `run_slot-spandec-strict-fan4-select.txt`,
`fan_geom_fan4-select_{2500,5000}_d6.{json,txt}` (3070 stream probe, 48 rows, depth 6),
`paired_select_5000.txt`. The train-side `fan/select_*` keys reached wandb from step 0
(the launch's open item).

**Final-val keys at 5,000.**

| key | value |
|---|---|
| `fan/stream_ce_k0..k3` | 4.5718 / 4.5703 / 4.6466 / 4.5870 (spread 0.076) |
| `fan/oracle_ce` | 4.4573 (0.113 below the best single stream) |
| `fan/mixed_ce` (the gate's argmax, the deployed write) | 4.5567 (0.014 below the best single) |
| `fan/gate_agree` | 0.332 |
| `fan/oracle_pick0` | 0.288 |
| `fan/stream_rank_t1` / `_t6` | 2.891 / 2.609 |
| `val/fan_stream_cos_t1` / `_t6` | −0.285 / +0.651 |
| `val/fan_mix_entropy` | 1.357 (ln 4 = 1.386) |

Selector regret `mixed − oracle` = 0.099 nats. The same decomposition at step 3250 read
0.102 / 0.009 / 0.34, so the picture was set by mid-run.

**Stream probe at 5,000 (3070).** Pass 1: rank 2.887, mean cos −0.29, `axis_cos` 0.099,
shared 0.20, norms 29.2 / 25.0 / 23.4 / 24.3, sign family `+---` 44 %. Pass 6: rank
2.587, cos +0.68, `axis_cos` 0.375, shared 0.87, norms 46.0 / 39.4 / 38.9 / 39.4, `+---`
48 %. The spread is made at passes 1–2 and the later passes pull the four back together
(cos −0.29 → +0.68), with stream 0 the largest.

**Train-side, steps 4000–5000 (50 rows).** `fan/select_pick0` mean 0.269 (min 0.171, max
0.430); shares k0..k3 mean 0.269 / 0.288 / 0.181 / 0.262, minima 0.174 / 0.187 / 0.078 /
0.169; `fan/select_agree` 0.323; `fan/select_gate_ce` 1.356 (ln 4 = 1.386); forced 0.050;
train-side oracle 0.116 below stream 0. `fan/select_gate_weighted / loss/total` after
step 1000: mean 0.124, max 0.144 at 3000.

**Depth sweep (480 rows).** 5,000: d1 4.4335, d2 4.4299, d3 4.4272, d6 4.4259, d9 4.4272;
K1−K6 +0.0076, K3−K6 +0.0013. 2,500: d1 4.7590, d6 4.7495 (+0.0095).

**Paired against the width partner (`paired_select_5000.txt`, the fan4 filing's block
bootstrap, 511,089 tokens, 500 blocks).** Depth 6: select 4.4259, pk4 4.3398,
`select − pk4` **+0.0861 [+0.0830, +0.0894]**; depth 1: +0.0921 [+0.0889, +0.0956].

**Not predicted, recorded.** `loop/core_gain_t0` climbed 1.35 → 2.35 (step 900) → 9.3
(2400) → max 20.7 (4799), last 17.5, while the epivol arm sat at 1.45–1.57 throughout;
preclip stayed 2–4 and the run did not detonate. The carry arms' scale mode had the
same shape at 500–3e5. Hypothesis, unscored: the argmin over K noisy streams favours the
stream that moved furthest, so winner-takes-all selects on variance and the gradient
rewards a larger first pass.

**Scoring.**

- **P-1: FAILS.** k2 sits 0.076 above the best stream (bar 0.05); k0, k1, k3 within 0.017.
- **P-2 (the falsifier): HOLDS.** Oracle 0.113 below the best single stream, five times
  the 0.022 bar and five times the mixture arms' 0.018.
- **P-3: FAILS.** The CE clause holds (0.014 > 0.005) and the agreement clause fails
  (0.332 < 0.40; chance given the pick shares is about 0.29).
- **P-4: HOLDS.** Rank 2.89 > 2.0; probe `axis_cos` 0.099 at pass 1 (< 0.5).
- **P-5: FAILS.** +0.0861 against a +0.005 bar; the arm is 0.086 nats behind its width
  partner (the mixture arm fan4 was +0.0073).
- **P-6: HOLDS.** pick0 0.269 in [0.15, 0.60]; smallest share 0.078 > 0.05.
- **P-7: HOLDS.** K1−K6 +0.0076 > +0.005; K3−K6 +0.0013 > +0.001.
- **P-8: FAILS on the share clause.** Rate 8,018 ≥ 6,000 and no tripwire hold; the gate
  term is 12–14 % of the loss after step 1000 against a 10 % bar (a 1.35-nat CE beside a
  9-nat total).

## Verdict

**Failure.** P-3 and P-5 fail, with P-1 and P-8; P-2, P-4, P-6 and P-7 hold. The arm did
what it was built for on the training side and not on the deployed side: winner-takes-all
made four individually readable streams with 0.113 nats of oracle value, and the gate that
must choose one stream before the span cashed 0.014 of it. The deployed one-stream write
is 0.086 nats behind a single stream of the same width.

## Updated hypothesis

1. Candidate value is real and cheap to make. One CE on a hard per-slot winner did in
   5,000 steps what three diversity terms could not: every stream has a reader and the
   oracle sits 0.113 below the best of them. Diversity terms are not the lever;
   responsibility is.
2. Selection before the span cannot cash it. The gate's agreement with the oracle is at
   chance and its CE is at ln 4: from the exit state alone, which stream the next span
   will favour is not predictable. That is the Jensen wall of the 2026-09-20 brainstorm:
   the target picks the mode, and a per-slot guess made before any of the span's tokens
   has no evidence. Selector regret 0.099 of 0.113.
3. The one-stream write pays twice: the guess, and a reader that sees each stream in
   about a quarter of the slots. Against pk4 the arm loses 0.086 nats at depth 6, twelve
   times the mixture arm's loss, so a hard commit is the wrong deployment even with the
   candidates in hand.
4. The other three streams are discarded at the write (`prefix_project` with
   `cells=None`), so nothing downstream can use them. The next arm keeps them: every
   stream written into its own prefix cell (the register's route) and the coda's own
   per-token attention as the selector, with this arm's winner-takes-all term as the
   responsibility (`2026-09-20-lxtul-fan4-all.md`, queued the same day). Its P-3 reads
   whether a selector with evidence closes the 0.099; its P-5 reads whether the write
   recovers the 0.086.
5. The first-iteration gain's climb under winner-takes-all is a new reading with no
   instrument on its cause; it plateaued at 20 here and did not detonate, and the all arm
   carries the same term, so its `loop/core_gain_t0` is read beside this one.
