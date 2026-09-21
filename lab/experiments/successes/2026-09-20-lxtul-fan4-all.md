# Planned: the fan with every stream written — the coda's own attention is the selector

Status: success

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

## Results

Run: 5,000 steps at `98e698a` on the 5090 through `run_recon.sh`, started 17:20 local
2026-09-20, DONE 18:50, tripwire HEALTHY (max 29.4 at step 219), final val_loss 4.4376,
RATE OK 7,096 tok/s at step 200, peak 19.14 GB. Artifacts in
`../results/2026-09-19-lxtul-fan4/`: `sweep_slot-spandec-strict-fan4-all_{2500,5000}.json`,
`worth_slot-spandec-strict-fan4-all_5000.json`, `slot_state_slot-spandec-strict-fan4-all_5000.json`,
`run_slot-spandec-strict-fan4-all.txt`, `fan_geom_fan4-all_5000_d6.{json,txt}` (3070 stream
probe, 48 rows, depth 6, from the `MORPH-all` worktree at `98e698a`), `paired_all_5000.{txt,json}`.
The train-side `fan/wta_*` keys reached wandb from step 0 (the launch's open item).

**Final-val keys at 5,000.**

| key | value |
|---|---|
| `fan/stream_ce_k0..k3` (stream i alone in its cell, the other three blank) | 4.4911 / 4.5013 / 4.5386 / 4.5362 (spread 0.048) |
| `fan/oracle_ce` | 4.3925 (0.099 below the best single stream) |
| `fan/mixed_ce` (all four cells written, the deployed read) | 4.4348 (0.056 below the best single stream) |
| `fan/mixed_ce − fan/oracle_ce` (selector regret) | 0.042 |
| `fan/oracle_pick0` | 0.322 |
| `fan/stream_rank_t1` / `_t6` | 2.820 / 2.060 |
| `val/fan_stream_cos_t1` / `_t6` | −0.295 / +0.057 |
| `val/fan_mix_entropy` | 1.386 (= ln 4; the mix is the mean, no gate) |

**Stream probe at 5,000 (3070, 2,573 slots).** Pass 1: rank 2.826, mean cos −0.29,
`axis_cos` 0.129, shared 0.22, norms 25.2 / 21.7 / 17.5 / 18.9, sign family `+---` 47 %.
Pass 6: rank 2.026, cos +0.09, `axis_cos` 0.259, shared 0.64, norms 56.0 / 26.2 / 18.1 /
24.7, `+---` 95 %. The same shape as select (spread at passes 1–2, the later passes pull
the four together with stream 0 growing), with a lower rank at pass 6 (2.03 vs 2.59).

**Train-side, steps 4000–5000 (50 rows).** `fan/wta_pick0` mean 0.350 (min 0.247, max
0.538); shares k0..k3 mean 0.347 / 0.267 / 0.215 / 0.171, per-row minima 0.256 / 0.193 /
0.108 / 0.082; forced 0.050; `fan/wta_ce` mean 4.373, the table's oracle 4.362, stream 0
alone 4.451. `fan/wta_weighted / loss/total` 0.33 throughout. `loop/core_gain_t0` max 9.92
at step 4680, last 6.53 (select reached 20.7; epivol 1.6): the winner-takes-all climb
is present and half the select arm's.

**Depth sweep (480 rows).** 5,000: d1 4.3105, d2 4.3072, d3 4.3060, d6 4.3056, d9 4.3061,
d12 4.3068, d16 4.3080; tokens K1−K6 +0.0049 [+0.0044, +0.0055], K3−K6 +0.0004 [+0.0003,
+0.0006]; the span decoder's own CE K1−K6 +0.0314 [+0.0302, +0.0326] (select +0.0574).
2,500: tokens K1−K6 +0.0024.

**Paired against the width partner (`paired_all_5000.txt`, 511,089 tokens, 500 blocks).**
Depth 6: fan4-all − pk4 **−0.0342 [−0.0367, −0.0316]**; depth 1: −0.0293 [−0.0320,
−0.0266]; every swept depth between −0.029 and −0.034. Select − pk4 at depth 6 was
+0.0861, so fan4-all − select is −0.120 by subtraction.

**Scoring.**

- **P-1: HOLDS.** The read cashes 0.056 against a 0.02 bar (the select gate cashed 0.014).
- **P-2: HOLDS.** Oracle 0.099 below the best single stream (bar 0.05; select 0.113).
- **P-3 (the arm's reason): HOLDS.** Regret 0.042 < 0.05 (select 0.099).
- **P-4: HOLDS.** Rank 2.82 > 2.0; probe `axis_cos` 0.129 at pass 1 (< 0.5).
- **P-5: HOLDS.** −0.0342 against a −0.005 bar; the interval excludes zero by 0.03.
- **P-6: HOLDS.** pick0 0.350 in [0.15, 0.60]; smallest share 0.082 > 0.05.
- **P-7: FAILS.** K1−K6 +0.0049 against +0.005 (the interval's upper end is +0.0055),
  K3−K6 +0.0004 against +0.001. Read as written, the bar is missed by 0.0001 and 0.0006;
  the arm's loop is as flat as every slot arm's (yardstick +0.002; select +0.0076).
- **P-8: HOLDS on the rate and tripwire clauses; the first clause is NOT SCORABLE as
  written.** `loss/ce_main` is `out["loss"]`, the whole training objective (13.27 here,
  with the span decoder's term and the WTA term inside it), not the token CE, so
  `fan/wta_ce − loss/ce_main` reads −8.9 and means nothing. The nearest honest reading is
  the final val: one stream alone in its cell (best 4.4911) against the four-cell read
  (4.4348), a gap of 0.056, inside the 0.30 bar. Rate 7,096 ≥ 5,000; no tripwire.

## Verdict

**Success** under the filing rule: P-3, P-4 and P-5 all hold, with P-1, P-2 and P-6; P-7
fails; P-8's first clause was miswritten and is recorded with its substitute reading.

The coda's own per-token attention, reading all four cells, recovers 0.056 of the 0.099
nats of oracle value that the select gate could not (0.014 of 0.113), leaves 0.042 of
selector regret, keeps the streams at rank 2.8, and the deployed arm is 0.034 nats
BELOW its width partner pk4 on paired rows where the select arm was 0.086 above it. The
standing rule applies to that last number: a paired 5k gap is context, not a ranking
(`short-horizon-ce-is-not-a-verdict`). What the arm is scored on, the K-curve, is flat:
tokens K1−K6 +0.0049, the same order as every slot arm. The write-all arm fixes the
reader and the selector; it does not make the loop earn depth.

## Updated hypothesis

1. The 2026-09-20 correction on the select filing stands and is now measured from the
   other side: the select deficit was NOT candidate value (unchanged at 0.099 here) and
   NOT the reader (every stream within 0.048 of the best). It was the deployment: a
   one-stream write chosen before the span, on a coda trained on the target's choice. Keep
   all four cells in the coda and the per-token read selects with the span's own
   evidence, at no extra pass at inference.
2. Selector regret 0.042 remains. The read is soft (attention over four cells inside a
   window that also holds the span's tokens), so part of the 0.042 is Jensen's gap on a
   soft mix and part is tokens early in a span that have no evidence yet. The per-token
   HARD read (top-1 over the four cells, the named next lever) tests the first; the
   worth profile by offset tests the second and is on file in `worth_…_5000.json`.
3. The streams still converge across the loop (pass-6 rank 2.03, stream 0's norm 56 vs
   18–26). Responsibility holds them apart where they are written, not where they loop;
   the later passes have no term that keeps four things four. That is the register's
   collapse pressure, slowed, not removed.
4. The loop is flat under the fan too (K1−K6 +0.0049 tokens; the span decoder's own read
   +0.031, a within-run signal that the slot's z does move with depth for the span
   decoder even when tokens do not). The fan family has answered the reader question and
   not the depth question. The depth lever is elsewhere (the hop staircase on the
   forced-through geometry, `hop-staircase-on-prev-reach1`).
5. `loop/core_gain_t0` climbs under winner-takes-all (9.9 here, 20.7 on select, 1.6 on
   epivol) and did not detonate. The select-gate arm (`2026-09-20-lxtul-fan4-select-gate.md`,
   running next) carries the same table and is read beside these two.
6. The P-8 lesson: name the key AND its scale in a clause. `loss/ce_main` is the whole
   objective on this tree; a token-CE clause must name a token-CE key.

**Correction, 2026-09-20 (after filing; Wolfe asked why the streams read low rank).**
`fan/stream_rank_t{t}` and the probe's `rank` are the participation ratio of the CENTERED
streams (`fan_stream_stats`, `morph/model/tul_fan.py`): four vectors minus their slot mean
sum to zero, so the ceiling for K = 4 is **3, not 4**. "2.82 of 4" above should read 2.82 of
a ceiling of 3, about 93 % of a regular simplex (whose pairwise cosine is −1/3; the arm
reads −0.29 at pass 1). The pass-1 reading is therefore at the volume term's own optimum.
The reading that IS low is pass 6: 2.03 of 3 with the shared fraction 0.22 → 0.64 and
stream 0's norm 25 → 56, while the select arm (balanced written shares) kept 2.59 at pass 6
with norms 46 / 39 / 39 / 39. Three correlated causes, not separated: the volume term is
charged after passes 1 and 2 only (`fan_repel_passes: 2`); the shared core contracts the
four states onto its dominant direction; winner-takes-all gives stream 0 most of the
gradient. P-4's bar (2.0) stands as written and holds. Named tests, not queued: the volume
term on every pass (`fan_repel_passes: 6`, one factor), and the core Jacobian's contraction
rate on the deviation subspace against the slot-mean direction.
