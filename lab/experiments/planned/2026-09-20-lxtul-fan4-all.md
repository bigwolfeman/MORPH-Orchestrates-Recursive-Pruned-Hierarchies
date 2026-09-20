# Planned: the fan with every stream written — the coda's own attention is the selector

Status: planned

Date: 2026-09-20 (frozen before any GPU step of the arm; a 12-step smoke of the config on
the Spark is the only run that may precede it). Arc: the LXTUL fan
(`2026-09-19-lxtul-fan4.md`, `2026-09-20-lxtul-fan4-epi.md`, `2026-09-20-lxtul-fan4-epivol.md`,
`2026-09-20-lxtul-fan4-select.md`). Direction: Wolfe, 2026-09-20 ("do the code/mux tokens
hold the other explorations? ... we should at least try that").

| arm | config | one factor against |
|---|---|---|
| `slot-spandec-strict-fan4-all` | `tul_slot_spandec_strict_fan4_all.yaml` | `slot-spandec-strict-pk4` (the ruler at `prefix_k` 4): K = 4 streams in the 4 prefix cells against ONE stream projected into 4 |
| | | `slot-spandec-strict-fan4-select`: the write (all K, 1:1) and the selector (the coda's attention, no gate) |

## Question

The select arm (`2026-09-20-lxtul-fan4-select.md`) settled two things at 5,000 steps.
Winner-takes-all responsibility makes candidate value: the oracle over the four streams
sits 0.113 nats below the best single stream, five times the mixture arms' 0.018, and
every stream has a reader (spread 0.076). And a gate that must choose BEFORE the span
cannot cash it: the gate's argmax stream recovers 0.014 of the 0.113, agreeing with the
oracle 33 % of the time, so the selector regret is 0.099 nats. The other three streams
are discarded at the write: the coda reads one cell and nothing else
(`transformer.py`, the fan site and `prefix_project` with `cells=None`).

**If every stream is written into its own prefix cell and the coda's attention picks per
token, with the span's own earlier tokens as evidence, does the coda recover the
oracle's value — and does winner-takes-all responsibility keep the streams from
collapsing the way the Thought Register's free read did?**

The Thought Register (`2026-09-13-arc-thought-register.md`, `thought-register-collapses`)
wrote four cells 1:1 through this exact route and read rank 1.24 of 4: with nothing
asking a cell to be individually useful, the four became one. This arm is that write plus
the one term the register did not have.

## Hypothesis

Two readings, and the arm can split them:

1. The select arm's regret is a causal-evidence limit (the Jensen wall in the 2026-09-20
   brainstorm): a per-slot choice before the span is a guess about the mode of the next
   span, and the target picks the mode. A per-token read inside the span has the mode's
   first tokens as evidence, so the coda's attention over four cells should cash a large
   part of the 0.113 and beat the hard commit.
2. The WTA term (K no-grad passes with stream i ALONE in its cell, the other cells blank;
   the winner replayed once with grad and its span CE charged) is enough responsibility
   to hold the streams apart under a free read. Against it: the register's collapse
   pressure (shared core, shared write, a span decoder that grades the MEAN) is all still
   there, and the term is one CE among three.

If (1) holds and (2) fails, the arm reads a good early number and a rank that falls; if
(2) holds and (1) fails, four distinct streams sit in the coda and the coda averages them
the way the softmax mixture did.

## Predictions (frozen)

All at 5,000 steps on the `fan/*_final` keys unless a training window is named; the
stream probe (`lab/divergence/fan_stream_probe.py`, 48 rows, depth 6) on the 3070; the
paired depth-6 read from the runner's sweep `tokens.npz` against the pk4 arm's, 480 rows.
Probabilities are the builder's.

- **P-1 (the read cashes more than the gate did).** `fan/mixed_ce` (the deployed all-K
  write) below the best single stream `min_i fan/stream_ce_k{i}` by more than **0.02**
  (the select gate: 0.014). **55 %.**
- **P-2 (candidate value stays).** `fan/oracle_ce` below the best single stream by more
  than **0.05** (select: 0.113; the mixture arms: 0.008 to 0.018). **60 %.**
- **P-3 (THE ARM'S REASON: the regret shrinks).** `fan/mixed_ce − fan/oracle_ce` below
  **0.05** nats (select: 0.099). **35 %.**
- **P-4 (no register collapse).** `fan/stream_rank_t1` above **2.0** of 4 AND the probe's
  `axis_cos` below **0.5** (register 1.24; select 2.89). **60 %.**
- **P-5 (against the width partner).** Paired depth-6 token CE on the same 480 rows:
  `fan4-all − pk4` below **−0.005** (the arm BEATS its width partner; fan4 − pk4 was
  +0.0073, select − pk4 is pending). **30 %.**
- **P-6 (the winner is not one stream).** Train-side `fan/wta_pick0` between **0.15 and
  0.60** over the last 1,000 steps and no `fan/wta_share_k{i}` below **0.05**. **60 %.**
- **P-7 (depth).** Token K1−K6 above **+0.005** and K3−K6 above **+0.001** (select
  +0.0076 / +0.0013). **40 %.**
- **P-8 (one stream alone stays readable, and the cost is paid).** Over the last 1,000
  steps, train-side `fan/wta_ce − loss/ce_main` (one stream alone in its cell against
  the full four-cell read, same batch) below **0.30** nats; step rate at step 200 at least
  **5,000** tok/s (select 8,018; the arm adds one coda pass with activations); no
  tripwire. **50 %.**

**Filing rule.** `successes/` only if P-3, P-4 and P-5 all hold: the arm exists to
recover the oracle causally, without the register's collapse, and to beat width. Any
other outcome is `failures/` with the per-clause table.

**Rejection of the reading.** If P-2 holds and P-3 fails, the coda's per-token read does
not select either and the Jensen-wall reading is withdrawn in favour of whatever the
oracle-versus-mixed table shows (a mixture of readers at eval is then the remaining
test). If P-4 fails, the WTA term is not enough responsibility and the next lever is the
per-token hard read (top-1 attention over the cells), not another diversity term.

## Method

**Build (commit in the queue line).** `tul.fan_mix: all` in `morph/model/tul.py`,
`tul_fan.py`, `transformer.py` (`_tul_fan_all`, `_tul_fan_stream_write`, the write's
`cells=` branch, the `fan_wta_*` fold), `tul_setup.py`, `train.py`. The config composes
`tul_slot_spandec_strict_fan4_epivol` (epivol repulsion 0.1 on passes 1..2, `prefix_k` 4,
`slot_cell_init: distinct`) with `fan_mix: all`, `fan_select_eps: 0.05`,
`fan_all_wta_lambda: 1.0`. The span decoder grades the MEAN of the cells (the register's
read; `spandec_reads_cells` stays off so the arm is one term from the register, not two).

**Deployed forward.** Cell i → prefix cell i through `W_prefix[i]` (`prefix_project(cells=)`,
the register's route); the coda reads all four; the eval `fan/mixed_ce` is this write.
"Stream i alone" everywhere (the oracle, the WTA table, the WTA replay) is cell i in its
cell with the other three blank (`_tul_fan_stream_write`), the deployed geometry minus the
losers; the select arm's single-source broadcast is left bit-identical on the mixing modes
(`tests/test_tul_fan_all.py`, `tests/test_tul_fan_select.py` 12 passed).

**Training.** ONE trainer on the 5090 through `run_recon.sh`, kind `slot`, sweeps
2500,5000, last 5000, smoke 1, outdir `/home/wolfe/morph-scratch/arc/results/2026-09-19-lxtul-fan4`,
bare config (no override, so the runner's own readouts run). 5,000 steps, seq 1024, batch
6, ramp 1,000, `norm_match`, the same seed as every fan arm.

**Scoring, in reading order.** (1) `fan/oracle_ce`, `fan/mixed_ce`, `fan/stream_ce_k{i}`
finals — P-1, P-2, P-3. (2) `fan/stream_rank_t1` and the 3070 stream probe — P-4. (3) the
paired depth-6 read against pk4 from `tokens.npz` — P-5. (4) train-side `fan/wta_*` and
`loss/ce_main` over steps 4000–5000 — P-6, P-8. (5) the runner's depth sweep at 5,000 —
P-7. (6) the RATE line and the tripwire — P-8.

**Amendments.** The Predictions section is frozen. If the method has to change it is
amended here with the date and the reason.

## Not verified before launch

- CPU only: `tests/test_tul_fan_all.py` 9 passed, `test_tul_fan_select.py` 12,
  `test_tul_fan.py` 21, `test_tul_fan_epi.py` 16, `test_tul_slot_register.py` 54,
  `test_tul_prefix_source.py` 54, `test_tul_setup_keys.py` 8 (one pytest at a time,
  `taskset -c 0,1`). The config composes through Hydra and `build_tul_runtime`.
- The WTA replay is NOT the oracle's number even at eps 0: the table scores slot s with
  stream i in every slot, the replay writes each slot its own winner, and the strict coda
  reads earlier slots' cells (0.005 nats apart on the CPU batch). Recorded, not a defect.
- `torch.compile` on the new method is disabled (`@torch.compiler.disable`), as the select
  arm's is; the surrounding graph is unchanged.
- The GPU smoke of the composed config: Spark, 2026-09-20, the runner's 12-step command
  (`training.steps=12 eval_every=6 n_eval_batches=2`), six runs in all. Runs 1–5 found
  and removed a memory cost that was NOT the replay: 26.3 GB with the replay unchecked,
  21.0 GB with it checkpointed as one segment, 21.7 GB checkpointed per coda block and per
  logit row (so the replay's activations were never the cost); a CUDA memory snapshot by
  allocating line put 6.0 GB on `prefix_project`'s broadcast matmul over `[B, S, K]`,
  which expands `W_prefix` to a `[B, S, K, C, C]` copy and keeps it for backward, once for
  the write and once for the replay. The cells branch now runs K plain matmuls. Run 6, on
  the code in the queue line: 21:54–21:59 UTC, exit 0, **peak 15.71 GB** (the select arm's
  smoke on the same day and machine: 17.63 GB; its shipped single-source write still
  carries 3.0 GB of the same copy, left untouched), final val `fan/mixed_ce` 11.1791,
  `fan/oracle_ce` 11.1713, `fan/oracle_pick0` 0.3135 (355 spans); the train-side
  `fan/wta_*` keys are proven present in the forward's output by the tests and are checked
  on the 5090 run's first logged steps. The smoke's step rate is not a reading (12 steps,
  compile warmup); P-8 reads the runner's RATE line.
