# Planned: is the plan EMPTY, or is the coda BYPASSING a usable one?

Status: planned
Instrument: `token_tax` + `plan_shuffled` in `lab/divergence/slot_path_worth.py`,
driven by `lab/divergence/reader_or_target.py`
Wolfe's redirect, 2026-08-28: the question is the coda bypassing z, not the history of spans.

## The gap in every panel so far

`plan_off` and `plan_shuffled` both ran with the coda's token path fully intact, because
`TULSlots.apply_token_dropout` returns its input unchanged when `training` is False. A coda
that can read a span's own tokens has no reason to consult a plan. So every number this
campaign has produced — plan worth 0.013–0.123, shuffle cost ~0.0001 — measures what the coda
**bothers** to use, not what it **could** use.

That single measurement is consistent with two very different worlds, and the campaign has
never separated them:

    z is EMPTY                 -> shuffling costs nothing because there is nothing there
    z is FULL but UNREAD       -> shuffling costs nothing because the coda never looks

`token_state_dropout` (0.15 in training) is the ONLY pressure that makes the coda consult the
plan, and it is switched off at eval. A note appended to `slot_path_worth.py` on 2026-08-25
described exactly this sweep. The code was never written. This is that code.

## Method

Force the dropout ON at eval and sweep the tax p ∈ {0.0, 0.5, 0.9, 1.0}. At each level
measure `full`, `no-plan` (z zeroed) and `plan-shuffled` (z permuted across slots). At
p = 1.0 every token state is `E_mask` — Bowman's inputless decoder, the extreme end of the
§3.4 arm sweep — so the coda has nothing but the plans and the positions.

The global RNG is reseeded before every condition, because the drop is sampled inside the
shipped `apply_token_dropout`. Without that, conditions would differ by which tokens were
masked as well as by the plan, and the comparison would be worthless.

`token_tax` flips the config value and forces the training branch of the SHIPPED function
rather than reimplementing the drop — the mask also has to zero the coda's x0 and bigram
injections at dropped positions, and a copy of that logic would drift from it silently.

Arms, at step 3000: `ctrlworth-s3` (65.1% specific, the POSITIVE CONTROL), `a1noaux-s1`
(0.4%), `tg2-s1` (0.3%), `tg3b-s1` (0.6%).

## Predictions (frozen 2026-08-28 13:15, before the sweep runs on any checkpoint)

**R1 (the tax bites).** `full` ce_main rises monotonically with p on every arm, and by
≥ 1.0 nats from p=0 to p=1. If starving the coda of every token state does not cost a nat,
the tax is not reaching the coda and nothing below is readable.

**R2 (the positive control reads its plan).** On `ctrlworth-s3` — the only arm with a
content-bearing plan — the shuffle cost at p=1.0 is ≥ 5x its cost at p=0.0. This is the
sanity check that the instrument can detect bypassing where bypassing is known to be
possible. If it fails, the sweep cannot answer the question for anyone.

**R3 (the aux-off arms have nothing to find).** On `tg2-s1`, `tg3b-s1` and `a1noaux-s1`, the
shuffle cost at p=1.0 stays below 0.05 nats. Reasoning: the completed 2x2 showed the
emit/plast losses are the only mechanism writing span-specific content, and all three arms
have them zeroed. There should be nothing for the tax to expose.

**R4 (zeroing always beats shuffling).** At every tax level and on every arm, the zero cost
is ≥ the shuffle cost. Zeroing removes the positions' content AND the plan; shuffling removes
only the correspondence. A shuffle that costs MORE than zeroing means the condition is
broken, not that the plan is harmful.

## Decision rule — this gates the flow-matching / diffusion path

- **R3 FAILS on any aux-off arm** (shuffle cost at p=1.0 ≥ 0.05) → **the coda is BYPASSING a
  usable plan.** The reader is the bottleneck, not the target. The fix is reader-side —
  raise `token_state_dropout`, inject the plan per-layer — and a training loss that writes
  MORE into z is well motivated because there is a reader that can be made to use it.
- **R3 HOLDS and R2 HOLDS** → the aux-off plans are genuinely EMPTY, and the instrument is
  proven able to detect otherwise. Writing more into z is then the ONLY lever that can work,
  and reader fixes cannot help because there is nothing to read. **This is the case that
  motivates flow matching**, and it must target span **i+1**, never span i — the DB campaign
  measured MORPH's SliceScaler putting 77% of training into autoencoding when the target was
  the current span.
- **R2 FAILS** → the instrument cannot detect bypassing even in the arm known to have
  content. Report that and do not read R3 either way.

Note what this rules out: if R3 holds, no amount of `token_state_dropout` tuning rescues the
TG2-family arms, because their objective never wrote anything to read. That retires the
"raise the collapse tax" arm as a fix on its own.

## Tests

Two cases in `tests/test_slot_seed.py`, both driving a REAL `MORPHTransformer` forward
rather than a stub — the rank bug earlier today came from a stub whose shape I chose myself.
`token_tax(0.0)` must reproduce the ordinary eval forward EXACTLY, and `token_tax(1.0)` must
move the loss by more than 0.2 nats. The magnitude assertion is deliberate: the tiny fixture
is randomly initialised and sits above uniform, so starving it pulls the loss DOWN. Asserting
"worse" passes on a trained model and fails here for a reason unrelated to the tax.
Verified to FAIL when `token_tax` is stubbed to a no-op. Full file: 26 passed.


## Results (filled 2026-08-28 15:00). The coda is NOT bypassing. The plans are EMPTY.

Shuffle cost (nats of ce_main) as the coda is starved of its token states:

    arm            aux   p=0.00   p=0.50   p=0.90   p=1.00   rise    zero cost @1.0
    ctrlworth-s3   ON    0.0096   0.0409   0.0800   0.0867   9.0x       0.0863
    a1noaux-s1     OFF   0.0001   0.0001   0.0004   0.0008  14.8x       0.0278
    tg2-s1         OFF  -0.0001   0.0003   0.0006   0.0013     —        0.1082
    tg3b-s1        OFF   0.0002   0.0002   0.0039   0.0104  45.6x      -0.0187

**R1 HELD.** `full` ce_main rises 4.47 → 7.47 on the control and similarly on every arm —
about 3 nats from p=0 to p=1. The tax reaches the coda.

**R2 HELD, and it is what makes the rest readable.** On the one arm with a content-bearing
plan the shuffle cost rises **9.0x**, and at p=1.0 specificity reaches **100.5%** — with no
token states left, shuffling costs exactly what zeroing costs, i.e. EVERY bit of the plan's
value is span-specific. The instrument detects bypassing where bypassing is possible.

**R3 HELD on all three aux-off arms.** Shuffle cost at p=1.0 is 0.0008 / 0.0013 / 0.0104,
all far under the 0.05 line. Starve the coda of every token state and shuffling the plans
STILL costs essentially nothing.

### The answer

**The coda is not bypassing a usable plan. In the aux-off arms there is nothing to bypass.**
The instrument is proven able to detect content — it finds it in ctrl-s3 and drives its
specificity to 100% under starvation — and it finds none in tg2, tg3b or a1noaux even at
full starvation.

So the earlier reading stands and is now causal rather than circumstantial: those arms'
objective never wrote anything into z, and **no reader-side fix can help them.** Raising
`token_state_dropout` is retired as a fix on its own — this sweep IS that fix, applied at
eval and taken to its limit, and it recovers 0.0008–0.0104 nats.

### What this says about the flow-matching / diffusion path

Per the decision rule, R2+R3 is exactly the case that MOTIVATES flow matching: writing more
into z is the only lever that can work, because there is nothing to read and no reader fix
helps. The shape is right, and it must target span **i+1**, never span i.

**But the sweep also measures the CEILING, and it is small.** ctrl-s3 is the only healthy
content-bearing plan in the campaign. Its plan is worth **0.0148 nats** in normal operation
and **0.0867 nats** with the coda fully starved. That is what a 65%-to-100% span-specific
plan buys at `seq_len 1024`. A flow-matching objective would have to beat a ~0.09-nat ceiling
to matter, and only 0.015 of that is reachable without also changing the reader.

That is the same conclusion the whole campaign keeps arriving at from different directions:
at this context length, with the tokens right there, a compressed plan has very little to do.

### R4 FAILED on tg3b-s1 at p=1.0, and the cause is arithmetic, not a broken condition

    tg3b-s1 @ p=1.0:  full 7.8090   no-plan 7.7902   shuffled 7.8194

Zeroing the plan IMPROVES the loss (−0.0187) while shuffling hurts it (+0.0104), so the
specificity RATIO reads −55.4%. The ratio's denominator has collapsed through zero and the
percentage is meaningless there. The condition is not broken — the same instrument produced a
coherent 100.5% on ctrl-s3 at the same tax.

Two real things are visible underneath: with every token masked, tg3b's plan is worth nothing
at all, and a WRONG-span plan is mildly HARMFUL — the coda has learned to read those
positions, so feeding them someone else's plan is worse than feeding them nothing. The same
effect appeared on the e_slot arms, where `no-loop` cost more than `no-plan`.

**Methodological note for anyone reusing this instrument: report the shuffle COST, not the
specificity FRACTION, whenever the zero cost is not comfortably positive.**
