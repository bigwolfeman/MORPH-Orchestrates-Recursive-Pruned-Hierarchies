# Planned: LXTUL-P rung P4, the relation half — K streams as K LINEAGES (`tul.fan_lineage`)

Status: failure

Date: 2026-09-21 (frozen before any GPU step of the arm). Arc: the LXTUL-P note
[`2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md`](../../../.agents/notes/proposed/architecture/2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md),
Part 2 change 4 / rung P4 and Alternatives item 6. Partner: the noise arm
[`2026-09-21-lxtul-fan4-all-noise.md`](2026-09-21-lxtul-fan4-all-noise.md), which this
config composes. LAST in the ladder by the note's own ordering.

## Question

Winner persistence on fan4-all is P(k*_{n+1} = k*_n) 0.3039 [0.2854, 0.3250] against
chance 0.2676 [0.2623, 0.2752] — an excess of +0.036, "a thin base for lineages". Stream
identities persist slightly more than chance while NOTHING in the model connects stream k
of one span to stream k of the next: the loop's cell relation lets a cell of slot n+1 read
every cell of every earlier slot, so the four streams re-mix at every span. If the
cross-slot half of that relation is narrowed to the own stream index — K separate channels
along the slot axis — does the persistence rise, and does the reader cash anything?

## The half that is NOT built, and why (decided at build, 2026-09-21)

The design's filtering posterior — `w_{n,k} ∝ w_{n-1,k} · exp(-l_{n,k}/tau)` from stream
k's span CE on the OBSERVED span n, resampling the lineages at span n+1's seed — is
circular in this forward, not merely awkward:

* `MORPHTransformer._tul_core` advances every slot's pass `t` TOGETHER: `h` is
  `[B, S*M, *carrier, C]` and one `_apply_core_step` moves all the cells, with the per-slot
  depth applied as a masked `torch.where(active, h_new, h)` at the foot of the loop. Slot
  n's EXIT state therefore does not exist when slot n+1's pass 1 runs.
* `l_{n,k}` is not a loop quantity at all. `_tul_fan_all` builds the `[B, S, K]` table from
  K no-grad CODA replays (`_tul_fan_stream_write` -> `_back_region`) AFTER `_tul_core` has
  returned. The weight that would gate pass 1 is a function of the coda's output, which is
  a function of the whole loop.

Three non-circular readings were considered and rejected: the PREVIOUS step's table (a
weight learned on another document); a second full forward (double cost, and it still
needs a loop before the loop); a slot-sequential loop (S times the sequential depth of the
shipped one, with S up to 64). Reading a span's own CE into its own seed would be the
fitted-z trap (`fitted-z-used-the-answer`) and is not done under any name. So this arm is
the relation ALONE, and the posterior waits for a forward that is sequential over spans.

## Hypothesis

A per-stream channel is the precondition a lineage needs and, on its own, probably not
worth CE: without reweighting nothing prefers the lineage that has been right, so the four
channels are four independent per-stream summaries of the row. The reading that would make
the rung worth continuing is a RISE in winner persistence well above the +0.036 excess —
that would say the channel carries an identity the coda can key on, and the missing piece
really is the weighting.

## Method

`morph/configs/tul_slot_spandec_strict_fan4_all_lineage.yaml`, one factor over the noise
arm: `tul.fan_lineage: relation`. Build: `morph/model/tul.py` (the key, its refusals and
the circularity note), `morph/model/transformer.py` (`slot_cell_relation(..., lineage=)`
narrows `blk` to `(same slot) OR (earlier slot AND same stream index)`; both callers — the
loop stage and the think-once stack — pass the SAME build-time bool),
`morph/training/tul_setup.py`, `tests/test_tul_fan_lineage.py` (7 tests: the mask by value
over every pair, the builder's default unchanged, off bit-identical, the narrowing
EXECUTES two-sided against an independently written mask, pads never keys, refusals,
compose). No new parameter, no new draw, one extra AND on a `[1, 1, S*M, S*M]` bool.

Same seed, packer, rows, 5,000 steps, batch 6. Readouts: the noise arm's, plus the WINNER
PERSISTENCE reading from `lab/divergence/fan_mixture_probe.py` on the same 48 rows.

## Predictions (frozen)

Baselines: fan4-all persistence 0.3039 [0.2854, 0.3250], chance 0.2676, shares
0.354 / 0.259 / 0.203 / 0.183. The noise arm's own numbers replace fan4-all's wherever
this arm is scored against its rung below. Seed spread 0.024 nats. Probabilities are the
builder's, reviewed by the orchestrator 2026-09-21 03:44 and adopted unchanged before any GPU step.

- **P-1 (stable).** 5,000 steps, tripwire HEALTHY. **85 %.** The change is a narrowing of
  an attention mask; it removes routes, it does not add scale.
- **P-2 (THE RUNG'S REASON: persistence rises).** Winner persistence above **0.36**, with
  its bootstrap interval clear of the noise arm's. **35 %.** Against it: the coda's read
  and the winner-takes-all loss are both per-span and neither rewards keeping an identity.
- **P-3 (the reader).** Paired depth-6 reader CE, lineage − noise, inside
  **[−0.024, +0.024]**. **60 %.** Residual: 30 % WORSE than +0.024 (the narrowing removes
  cross-stream routes the loop was using), 10 % better.
- **P-4 (the rank does not fall).** `fan/stream_rank_t6` within **±0.3** of the noise
  arm's. **65 %.** Blocking cross-stream reads should, if anything, slow the mixing.
- **P-5 (stream shares stay uneven).** The winner share of the most-used stream stays
  above **0.30** (fan4-all: 0.354). **70 %.** Nothing here equalises the streams.
- **P-6 (cost).** RATE within **3 %** of the noise arm's. **90 %.**

## Binding

If **P-2 holds**: the channel carries an identity and the missing piece is the weighting.
The next build is the sequential-over-spans forward (or an explicit two-stage forward)
that makes the posterior causal, and it is a real architecture change with its own note —
not a knob.

If **P-2 fails and P-3 holds**: a per-stream channel costs nothing and buys nothing.
Rung P4 closes as "lineages need the weighting, and the weighting needs a different
forward"; record it in the LXTUL-P note and do not build the sequential loop for this.

If **P-3 lands in its 30 % tail** (worse): the cross-stream cross-slot routes were
load-bearing, which is itself a finding about what the fan's streams are doing — read the
oracle and the per-stream span CE spread before calling it a loss.

## Not verified before launch

- No GPU step. Wall clock and memory unmeasured.
- The think-once stack (`tul.cond_layers`) also takes the narrowed relation, and no arm in
  this panel builds one, so that path is UNEXERCISED by any test beyond the shared builder.
- The interaction with the noise arm is a conjunction: if the noise arm fails P-1 or P-2
  this arm's base is gone and it must be re-cut over whatever base survives.
- No measurement says the relation narrowing is what changes persistence rather than the
  general loss of cross-stream routes; the arm cannot separate those two.

## Results

Run: 5,000 steps at `36a571d` on the 5090 through `run_recon.sh`, START 10:52 local
2026-09-21, DONE 12:18, tripwire HEALTHY (`preclip/total` max 52 at step 220), final
val_loss 4.4479, RATE 7,318 tok/s at step 200 (7,292 at 4800), peak 19.14 GB. Artifacts in
`../results/2026-09-19-lxtul-fan4/`: `sweep_slot-spandec-strict-fan4-all-lineage_{2500,5000}.json`,
`worth_..._5000.json`, `slot_state_..._5000.json`, `run_slot-spandec-strict-fan4-all-lineage.txt`,
`wandb_series_lineage.json`, `loop_norms_lineage_vs_noise.txt`, `fan_geom_lineage_5000_d6.{json,txt}`
and `fan_mixture_lineage_5000_d6.{json,txt}` (Spark, worktree `MORPH-0921` at `82868b4`,
the same 48 rows), `paired_lineage_vs_noise_5000.json`, `paired_lineage_vs_fan4all_5000.json`.

**The base is gone, as the prereg named.** The noise arm failed its P-1/P-2 clause pair
on P-2 (`../failures/2026-09-21-lxtul-fan4-all-noise.md`): its K draws were escaped by a
scale growth of the entry state. This arm composes that config and shows the same escape,
smaller: `loop/in_norm_t0` **1,365 → 2,454 → 4,752 → 7,720 → 10,317** (noise 15,417 at the
end), streams at cosine +0.99 from pass 0. Everything below is read as "what does the
per-stream channel do on a collapsed fan", not as rung P4.

**Last-val keys (step 4,750; wandb).**

| key | lineage | noise (rung below) | fan4-all |
|---|---|---|---|
| `fan/oracle_ce` | 4.3613 | 4.3663 | 4.3925 |
| `fan/mixed_ce` | 4.4016 | 4.4113 | 4.4348 |
| regret | 0.040 | 0.045 | 0.042 |
| `fan/stream_rank_t1` / `_t6` | **2.407 / 2.085** | 1.542 / 1.323 | 2.820 / 2.060 |
| `val/fan_stream_cos_t1` / `_t6` | +0.987 / +0.982 | +0.991 / +0.984 | −0.30 / +0.06 |
| `loop/core_gain_t0` max | 1.78 | 1.74 | 9.92 |

**Stream probe at 5,000 (Spark, 2,573 slots).** Pass 0: cos +0.990, rank **2.860**, norms
333 / 345 / 346 / 338, `++--` 33 %. Pass 1: rank 2.370, `+---` 43 %. Pass 6: cos +0.981,
rank **2.032**, norms 401 / 430 / 431 / 421, `+---` 44 %. The noise base reads 1.827 →
1.263 on the same rows. Inside a deviation that is 1 % of the state, the per-stream
channel keeps the four deviations spread at fan4-all's rank (2.31 → 2.03), where the
all-to-all relation let them merge.

**Mixture probe at 5,000 (Spark, same rows).** Oracle 4.2095, deployed 4.2515 (agreement
8e-8 / 3e-8); deployed − prefix mixture −0.0092 [−0.0114, −0.0071]; deployed − oracle
+0.0419; best single 0.098 above the oracle; **winner persistence 0.3002 [0.2776, 0.3217]
vs chance 0.2558** (excess +0.044; noise 0.2708 vs 0.2525, +0.018; fan4-all 0.3039 vs
0.2676, +0.036); shares 0.219 / 0.228 / 0.315 / 0.238.

**Depth sweep (480 rows).** 5,000: d1 4.3143, d6 4.3113, d16 4.3132; tokens K1−K6
**+0.0029 [+0.0025, +0.0034]**, K3−K6 −0.0001. **Paired at depth 6:** lineage − noise
**+0.0040 [+0.0018, +0.0063]**; lineage − fan4-all +0.0057 [+0.0036, +0.0079]. Worth: zero
0.1989, shuffle 0.1962, wrong_seed 0.0878.

**Scoring.**

- **P-1: HOLDS.** 5,000 steps, HEALTHY.
- **P-2 (the rung's reason): FAILS.** Persistence 0.300 against 0.36, its interval
  overlapping the noise arm's [0.249, 0.294] by 0.017.
- **P-3: HOLDS.** +0.0040 [+0.0018, +0.0063], inside ±0.024.
- **P-4: FAILS on its letter, in the good direction.** rank_t6 2.085 is 0.76 ABOVE the noise
  arm's 1.323, outside ±0.3. The narrowing did not let the rank fall; it held it.
- **P-5: HOLDS.** Top share 0.315 > 0.30.
- **P-6: HOLDS.** 7,318 within 3 % of 7,295.

## Verdict

**Failure** on the rung's reason (P-2), 4 of 6 hold. Binding branch: "P-2 fails and P-3
holds: a per-stream channel costs nothing and buys nothing. Rung P4 closes as 'lineages
need the weighting, and the weighting needs a different forward'". The channel does
one measurable thing, holding the centred rank of the (collapsed, 1 %-scale) deviations at
2.0 instead of 1.3 through the loop, and that rank buys the reader nothing (+0.004 paired,
inside the spread) and the winners nothing (persistence +0.044 over chance against
+0.018 and +0.036). The sequential-over-spans forward is NOT built for this. Two arms in a
row now read a regret of 0.040–0.045 and an oracle 0.10 below the best single on
streams at cosine 0.99: the oracle-gap floor named in the noise filing stands.

## Updated hypothesis

A per-stream channel along the slot axis is a precondition, not a lever: with nothing
preferring the lineage that has been right, K channels are K parallel summaries and the
coda keys on none of them. Lineages become a live question only on a fan whose streams
are distinct in the first place (the volume term, or a per-pass job), and only with a
causal weighting, which this forward cannot compute. Rung P4 is closed on this tree.
