# Planned: the select arm writes the gate's own pick — closing the train/eval mismatch

Status: planned

Date: 2026-09-20 (frozen before any GPU step of the arm; a 12-step smoke of the config on
the Spark is the only run that may precede it). Arc: the LXTUL fan
(`2026-09-19-lxtul-fan4.md`, `2026-09-20-lxtul-fan4-epi.md`, `2026-09-20-lxtul-fan4-epivol.md`,
`../failures/2026-09-20-lxtul-fan4-select.md`, `2026-09-20-lxtul-fan4-all.md`). Direction:
Wolfe, 2026-09-20, on the select filing ("I feel like this cant transfer to inference at
all, and may be part of our over all problem ... this sounds easily resolved. ... go,
queue it behind write-all").

| arm | config | one factor against |
|---|---|---|
| `slot-spandec-strict-fan4-select-gate` | `tul_slot_spandec_strict_fan4_select_gate.yaml` | `slot-spandec-strict-fan4-select`: `tul.fan_select_write` oracle → anneal (1500 steps). Same eps 0.05, gate weight 1.0, epivol term, seed, rows, recipe. |
| | | `slot-spandec-strict-pk4` (the ruler at `prefix_k` 4): the width partner every fan arm is paired against |

## Question

The select arm (`../failures/2026-09-20-lxtul-fan4-select.md`, correction of 2026-09-20)
trained its coda on the TABLE's winner: in 95 % of slots the stream written at train was
chosen by reading the span's target through K coda passes, and at eval the gate's guess
was written instead. The coda was trained on one input distribution and deployed on
another, the hindsight-fitted-z trap (`fitted-z-used-the-answer`) in a new place. Its
deployed write sat 0.086 nats behind the width partner at depth 6 (paired, 511k tokens),
the gate agreed with the table 33 % of the time, and the selector regret
`mixed_ce − oracle_ce` was 0.099 of an oracle value of 0.113. None of that separates a
gate that cannot predict from a coda that never saw what the gate picks.

**When the training pass writes the stream the GATE picks, so the coda's forward is the
same at train and at eval, does the deployed one-stream write recover against the width
partner, and does the gate's pick become a stream the coda reads well?**

## Hypothesis

`tul.fan_select_write: anneal` (`morph/model/tul.py`, `MORPHTransformer._fan_select_written`).
The table of K no-grad passes still runs every train step, but only as the gate's LABEL.
What the training pass writes moves, per slot, from the table's winner to the gate's argmax
with probability `step / 1500` (clamped at 1; scheduled sampling), with the eps 0.05 random
write on top under every mode. After step 1500 the coda is trained on exactly the stream it
will be deployed on, and the gate keeps learning the table's winner from a coda that now
reads the gate's picks.

If the 0.086 deficit was the mismatch, the coda adapted to the gate's picks reads them
close to what it read the winner at, and the paired gap against pk4 falls to the mixture
arms' range. The price is a closed loop: the gate picks, the coda learns those picks, the
table then favours the streams the coda reads best, which are the ones the gate picks. That
loop can collapse the fan onto one stream (rank falls, one written share dominates), in
which case the arm is pk4 with a dead fan and the honest closing is that the select
family cannot both explore and deploy. The 1500-step anneal exists to let the streams be
individually readable BEFORE the loop closes; whether that is enough is the risk.

## Predictions (frozen)

At the FINAL val (`*_final` keys at step 5000; the periodic val at 4750 is the fallback),
scored by `../results/2026-09-19-lxtul-fan4/fan_score.py`, the stream probe and the
paired block bootstrap of the fan4 filing. Under `select`, `fan/mixed_ce` is the SHIPPED
write, the gate's argmax stream written alone. Train-side series are wandb `fan/select_*`
rows; the new keys are `fan/select_write_p_gate`, `fan/select_write_from_gate`,
`fan/select_written_ce`, `fan/select_written_agree`, `fan/select_written_share_k{i}`.

- **P-1 (THE MECHANISM: the mismatch closes).** Train-side regret, the mean of
  `fan/select_written_ce − fan/select_oracle_ce` over the last 1,000 steps, and eval-side
  regret, `fan/mixed_ce − fan/oracle_ce` at the final val, are within **0.02** nats of
  each other. On the select arm they were about 0.006 (eps only) against 0.099. **70 %.**
- **P-2 (THE ARM'S REASON: the deployed write recovers).** Paired depth-6 token CE on the
  same 480 rows: `select-gate − pk4` below **+0.030** (select read +0.086 [+0.083, +0.089];
  the mixture arm fan4 +0.007; the ruler's seed spread is 0.024). **45 %.** Recorded beside
  it, not a clause: the paired `select-gate − select` gap, expected below −0.03.
- **P-3 (the gate's pick is well read).** `fan/mixed_ce` below `min_i fan/stream_ce_k{i}`
  by more than **0.02** nats (select: 0.014). The forced single-stream table is now the
  off-policy reading and the gate's pick the on-policy one; this is the first select arm
  where the deployed write should beat every forced stream by a margin. **40 %.**
- **P-4 (the oracle's headroom over the deployed write shrinks).** `fan/mixed_ce −
  fan/oracle_ce` below **0.05** (select 0.099). If the coda specialises to the gate's picks,
  the table's argmin loses value that the gate cannot reach. **55 %.**
- **P-5 (no collapse onto one stream).** Over the last 1,000 steps, no
  `fan/select_written_share_k{i}` above **0.70** and none below **0.05**; and
  `fan/stream_rank_t1` above **2.0** of 4 at the final val (select 2.89). **50 %.** If this
  fails with P-2 holding, the arm is pk4 with a dead fan and P-2 is width, not selection.
- **P-6 (depth).** Token K1−K6 above **+0.005** and K3−K6 above **+0.001** (select +0.0076
  / +0.0013). **40 %.**
- **P-7 (cost and stability).** Step rate at step 200 ≥ **6,000** tok/s (select 8,018; the
  passes are the same); no tripwire; `loop/core_gain_t0` maximum below **25** (select
  20.7, epivol 1.6); `fan/select_gate_weighted` below **15 %** of `loss/total` after step
  1000 (select 12 to 14 %). **60 %.**
- **P-8 (the schedule is what it says).** `fan/select_write_p_gate` reads exactly 1.0 from
  step 1500 on, and `fan/select_write_from_gate` averages **0.94 to 0.96** over the last
  1,000 steps (1 − eps, the forced slots keep their random stream). **90 %.**

**Filing rule.** Success iff P-1, P-2 and P-5 all hold. P-2 without P-5 files as a failure
with the width reading; P-1 and P-5 without P-2 file as a failure that closes the select
family on the deployed write (the mismatch was not the deficit).

**NOT predicted:** the arm's absolute CE, other anneal horizons, `fan_select_write: gate`
from step 0 (no anneal), other eps, transfer to math or Sudoku, what the write-all arm
reads (its prereg stands on its own).

## Method

Trained on the 5090 through `run_recon.sh` (seq 1024, batch 6, 5000 steps, sweeps 2500
and 5000, kind `slot`, outdir `2026-09-19-lxtul-fan4`), queued as the second line of
`recon_arms.txt` behind `slot-spandec-strict-fan4-all` and ahead of the bootstrap and vq8
arms. The runner's readouts run the bare config, which is this arm's config (no override),
so its sweeps, worth profile and state probe are the runner's own. Readings in order:
(1) P-8 on the first logged steps and again at 1500 (the schedule is the one thing that
can silently not happen); (2) P-1 from the train series against the final val (the
mechanism); (3) P-2 from the two sweeps' `tokens.npz` against pk4 AND against select, the
fan4 filing's block bootstrap; (4) P-3 and P-4 from the final-val fan keys; (5) P-5 from
the written shares and the 3070 stream probe (`lab/divergence/fan_stream_probe.py`, 48
rows, depth 6, launched from a worktree at this arm's commit); (6) P-6 from the K-curve;
(7) P-7 from the rate line, the tripwire and `loop/core_gain_t0`.

Instruments: `tests/test_tul_fan_select_write.py` (8 passed, CPU) proves that `oracle`
is the filed arm bit for bit (same loss, no RNG draw, no write stats), that `gate` writes
the gate's argmax on every non-forced slot and keeps a forced slot random, that the loss
under `gate` differs from the loss under `oracle` on the same table, that `anneal`
follows the `fan_select_step` buffer (0 → winner, 500/1000 → a mixture, past the horizon
→ gate), that the eval write is the gate's argmax under every mode, the refusals, and
that the config composes to `anneal` over 1500. A sabotage check (writing `choice`
instead of `written`) failed exactly the loss-differs test. `tests/test_tul_fan_select.py`
12, `test_tul_fan.py` 21, `test_tul_fan_all.py` 9, `test_tul_setup_keys.py` 8,
`test_checkpoint_compat.py` 5 passed at the same tree.

## Not verified before launch

- The GPU smoke of the composed config: see the smoke record appended below before the
  commit that carries this file.
- Whether the gate's label should be the table's winner WITH the eps random picks (as the
  select arm did; kept unchanged here, one factor) or the clean argmin. Not varied.
- Whether the 1500-step horizon is long enough for the streams to become individually
  readable before the loop closes. P-5 reads the outcome, not the horizon.
- The train-side `fan/select_written_*` keys are proven present in the forward's output
  by the tests; their wandb rows are read on the 5090 run's first logged steps.

**Smoke record (Spark, `~/smokes/fan4-select-gate/run.sh`, the runner's 12-step command,
2026-09-20).** Two runs. Run 1, 22:12:38 to 22:17:27 UTC, exit 0, at the tree before the
`select_written_share_k{i}` stats were added. Run 2, 22:18:21 to 22:23:07 UTC, exit 0, at
the committed tree (`transformer.py` md5 `6158d58f` on both hosts), peak 17.63 GB at step
0 (the select arm's smoke read 17.63), build banner `mix='select' ... train
WRITE='anneal' over 1500 steps` and the trainer line `[fan] select write anneals oracle
-> gate over the first 1500 steps`, final val `val_loss` 11.1743, `fan/gate_agree`
0.4789, `fan/mixed_ce` 11.1730, `fan/oracle_ce` 11.1571 (355 spans). At 12 steps the
schedule is 0.008 of the way to the gate, so the smoke exercises the buffer read and the
per-slot draw, not the gate write itself; the train-side `fan/select_write_*` rows are
in the offline wandb run and are read on the 5090 run's first logged steps. The GPU was
otherwise idle; the step rate of a 12-step smoke is not a reading.
