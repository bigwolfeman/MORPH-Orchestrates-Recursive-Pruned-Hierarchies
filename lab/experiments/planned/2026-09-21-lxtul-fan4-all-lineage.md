# Planned: LXTUL-P rung P4, the relation half — K streams as K LINEAGES (`tul.fan_lineage`)

Status: planned

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
