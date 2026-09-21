# Planned: the select arm writes the gate's own pick — closing the train/eval mismatch

Status: failure

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

## Results

Run: 5,000 steps at `57a0c3a` on the 5090 through `run_recon.sh`, started 19:07 local
2026-09-20, DONE 20:26, tripwire HEALTHY (max 42.3 at step 241), final val_loss 4.4922,
RATE OK 8,288 tok/s at step 200, peak 21.06 GB. Artifacts in
`../results/2026-09-19-lxtul-fan4/`: `sweep_slot-spandec-strict-fan4-select-gate_{2500,5000}.json`,
`worth_slot-spandec-strict-fan4-select-gate_5000.json`,
`slot_state_slot-spandec-strict-fan4-select-gate_5000.json`,
`run_slot-spandec-strict-fan4-select-gate.txt`, `fan_geom_fan4-select-gate_5000_d6.{json,txt}`
(3070 stream probe, 48 rows, depth 6, from the `MORPH-gate` worktree at `57a0c3a`),
`paired_select_gate_5000.{txt,json}`. The train-side `fan/select_write_*` and
`fan/select_written_*` keys reached wandb from step 0 (the launch's open item).

**The schedule, as logged.** `fan/select_write_p_gate` 0.000 at step 0, 0.493 at 740,
1.000 from step 1500 on; `fan/select_write_from_gate` 0.016 at step 20, 0.515 at 740,
0.957 at 1500, mean 0.952 (0.926 to 0.977) over steps 4000 to 5000.

**Final-val keys at 5,000.**

| key | value |
|---|---|
| `fan/stream_ce_k0..k3` | 4.4894 / 4.5117 / 4.5189 / 4.5297 (spread 0.040) |
| `fan/oracle_ce` | 4.4704 (0.019 below the best single stream; select 0.113, fan4-all 0.099) |
| `fan/mixed_ce` (the gate's argmax written, the deployed write) | 4.4894 (= `stream_ce_k0`: the gate writes stream 0) |
| `fan/mixed_ce − fan/oracle_ce` (eval-side selector regret) | 0.019 (select 0.099) |
| `fan/gate_agree` / `fan/oracle_pick0` | 0.514 / 0.514 |
| `fan/stream_rank_t1` / `_t6` | 2.857 / 1.999 |
| `val/fan_stream_cos_t1` / `_t6` | −0.306 / +0.333 |
| `val/fan_mix_entropy` | 1.275 (ln 4 = 1.386) |

**Train-side, steps 4000 to 5000 (50 rows).** Written shares k0..k3 mean 0.954 / 0.016 /
0.015 / 0.015 (k0 min 0.928, max 0.991); the TABLE's winner shares 0.435 / 0.222 / 0.172 /
0.171 (the gate's label stays spread while its pick does not); `fan/select_written_ce`
mean 4.462, `fan/select_oracle_ce` 4.432, train-side regret **0.030**; `select_written_agree`
0.437, `select_agree` 0.447, `select_gate_ce` 1.288 (ln 4 = 1.386); forced 0.048;
`fan/select_gate_weighted / loss/total` after step 1000 mean 0.120, max 0.142.
`loop/core_gain_t0` max 11.41 at step 2922, last 4.29 (select 20.7, fan4-all 9.9).

**Stream probe at 5,000 (3070, 2,573 slots).** Pass 1: rank 2.863, mean cos −0.31,
`axis_cos` 0.151, shared 0.16, norms 17.8 / 15.9 / 15.1 / 15.3, sign family `+--+` 46 %.
Pass 6: rank 1.986, cos +0.38, `axis_cos` 0.256, shared 0.75, norms 27.1 / 14.9 / 12.4 /
15.3, `+---` 64 %. The streams are spread at pass 1 (the epivol term) and stream 0 alone
grows through the loop; the coda reads only it.

**Depth sweep (480 rows).** 5,000: d1 4.3632, d2 4.3612, d3 4.3605, d6 4.3603, d9 4.3606,
d12 4.3611, d16 4.3618; tokens K1−K6 +0.0030 [+0.0026, +0.0033], K3−K6 +0.0002 [+0.0001,
+0.0003]. 2,500: d1 4.6912, d6 4.6892, K1−K6 +0.0020.

**Paired (`paired_select_gate_5000.txt`, 511,089 tokens, 500 blocks).** Depth 6:
select-gate − pk4 **+0.0205 [+0.0183, +0.0227]**; depth 1: +0.0234. Against the select
arm's own depth-6 sweep: **−0.0656 [−0.0689, −0.0623]**.

**Scoring.**

- **P-1 (the mechanism): HOLDS.** Train-side regret 0.030, eval-side 0.019; gap 0.011 <
  0.02 (select: about 0.006 against 0.099). The coda is deployed on what it was trained on.
- **P-2 (the arm's reason): HOLDS.** +0.0205 against a +0.030 bar; the arm recovered
  0.066 of the select arm's 0.086 deficit (recorded: select-gate − select = −0.0656).
- **P-3: FAILS.** The deployed write equals stream 0 alone (0.000 against > 0.02): the
  gate writes stream 0 in 95 % of slots, so "the gate's pick" and "the best forced
  single stream" are the same stream.
- **P-4: HOLDS, by collapse.** Regret 0.019 < 0.05, because the oracle's headroom over
  stream 0 fell from 0.113 to 0.019: the coda specialised to the one stream it is fed.
- **P-5 (no collapse): FAILS.** Written share of stream 0 is 0.954 over the last 1,000
  steps against a 0.70 ceiling; the other three sit at 0.015 each (the eps writes).
  Rank at pass 1 holds (2.86 > 2.0) because the epivol term still spreads the streams
  where they are made; the READ collapsed, not the geometry.
- **P-6: FAILS.** K1−K6 +0.0030 against +0.005; K3−K6 +0.0002 against +0.001.
- **P-7: HOLDS.** 8,288 tok/s; no tripwire; gain max 11.4 < 25; gate share max 14.2 % < 15 %.
- **P-8: HOLDS.** 1.0 from step 1500; from-gate mean 0.952 in [0.94, 0.96].

## Verdict

**Failure** under the filing rule (P-5 fails; P-1 and P-2 hold). The prereg's own clause
for this outcome applies: "P-2 without P-5: the arm is pk4 with a dead fan and P-2 is
width, not selection." The mismatch closed exactly as predicted (P-1), the deployed
write recovered 0.066 of the select arm's 0.086 deficit (P-2), and the closed loop of
gate, coda and table settled on one stream within 500 steps of the gate write turning
on. What is left is the strict ruler at prefix_k 4 with stream 0 as its state, paying
+0.020 nats against pk4 for the K-pass table, the eps writes into a coda that no longer
reads them, and the gate term.

## Updated hypothesis

1. The select family's deployed write cannot both explore and deploy. A before-the-span
   commit has to be TRAINED on its own picks to transfer (P-1 proves that transfer), and
   training on its own picks removes the reader's reason to keep any other stream. The
   two arms bracket it: select (oracle write) keeps four streams and cannot deploy;
   select-gate (gate write) deploys and keeps one. The select family closes on the
   deployed write; this note's alternatives (a clean argmin label, a longer anneal,
   Gumbel through the gate) all sit inside the same loop and are not queued.
2. The write-all arm (`../successes/2026-09-20-lxtul-fan4-all.md`) is the shape that
   escapes it: no commit before the span, every stream stays in the coda, selection per
   token. Its 0.042 regret is the number to work on; the per-token HARD read is the
   named lever there.
3. The table's oracle headroom is a reader property, not a stream property. Same streams
   (rank 2.86 at pass 1 here, 2.89 on select), same term, and the headroom is 0.019 or
   0.113 depending on what the coda was trained to read. `fan/oracle_ce − best single`
   measures the coda's breadth, and a small value can mean collapse as easily as copies.
4. Winner-takes-all does not move `loop/core_gain_t0` the same way twice (20.7 select,
   9.9 all, 11.4 here); none detonated. Still no instrument on its cause.
5. The loop is flat on every fan arm (+0.003 to +0.008 tokens). The fan family has not
   touched depth.
