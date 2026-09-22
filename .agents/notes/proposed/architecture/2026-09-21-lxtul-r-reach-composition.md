# Agent Note: LXTUL-R, the reach composition (make the slot loop's depth a job it cannot skip)

Status: proposed

Date: 2026-09-21. Decided with Wolfe in session after the LXTUL-P ladder closed
([`2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md`](2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md),
nine arms, one success on width, no depth). This note is the design document for what
comes next. It is a composition of measured pieces plus ONE small build (the coda's token
reach dial). Every knob it names exists in the tree today unless the row says "build".

## Problem

The slot loop earns no depth on any single-stream or K-stream arm (token K1−K6 inside
[−0.0002, +0.005] on fourteen arms; K3−K6 at zero). The LXTUL-P ladder tested the passes
themselves (per-pass jobs, state levers, entry noise, no prelude, lineage, denoising) and
closed every one with a mechanism. The remaining explanation is upstream of the passes, and
it has three parts, each measured:

1. **The loop's forced share is small.** Depth is earned in proportion to the loss share
   that has no shallower route and that pass 1 cannot satisfy alone. Three points on one
   line: the plain loop with a noise entry earns 0.185 (everything must go through the
   loop), the plain loop with a prelude entry 0.033 (the prelude took the easy part), the
   slot loop 0.002 (tokens bypass it by design; it is a side channel). The whole slot
   channel is worth 0.187 nats at 5k with cells zeroed (pk8 0.195, fan4-all 0.196), 0.76
   nats on the first token after a boundary and 0.09 by 16 tokens in
   (`lab/experiments/results/2026-09-12-strict/worth_*`, `2026-09-13-register/worth_*`,
   `2026-09-19-lxtul-fan4/worth_*`). K1−K6 is 1 % of that channel.
2. **A 4x horizon does not move it.** Plain norm-match-20k K1−K6 0.136 / 0.156 / 0.167 /
   0.170 at 5k / 10k / 15k / 20k; the strict slot loop 0.0001 / 0.0011 / 0.0012 / 0.0014 on
   the same 480 rows. Horizon is not the lever.
3. **The only operation that consumes depth here is relay across slots.** Amendment 3 of
   the strict panel (`loop_reach: 1` + `tg_coda_prefix_reach: prev`, arm
   `slot-spandec-strict-prev-reach1`) is the one change that ever moved the slot loop's
   token K-curve (+0.0163, K3−K6 +0.0042, at CE parity with strict), because content h
   spans back can only arrive at pass h−1. Its two measured defects are the design's
   central problem: (a) loop-carried content decays about a third per pass under the
   cell's later passes (a planted copy two spans back: 0.148 → 0.108 → 0.088 → 0.080;
   [`failures/2026-09-19-hop-distance-plateau-and-dilution.md`](../../../../lab/experiments/failures/2026-09-19-hop-distance-plateau-and-dilution.md)),
   and re-supplying it as a sum or a gate replaced the pass-1 read, diluted the far hops
   and ran unbounded (carry RMS 0.5 → 75;
   [`failures/2026-09-19-loop-carry-prev-reach1.md`](../../../../lab/experiments/failures/2026-09-19-loop-carry-prev-reach1.md));
   (b) Wolfe's objection on the object: under reach, z holds HISTORY, and z should hold
   the present next thought that needs decoding
   ([`2026-09-19-loop-carry-reinjection.md`](2026-09-19-loop-carry-reinjection.md),
   [`2026-09-13-tul-as-memory.md`](2026-09-13-tul-as-memory.md)).

The other pieces are solved and filed: the reader must be live and read every cell per
token ([`2026-09-20-fan-write-all-wta.md`](2026-09-20-fan-write-all-wta.md): the per-token
read cashes 0.056 where a before-the-span selector cashed 0.014); the K streams stay apart
under the within-slot volume term and gain candidate value under winner-takes-all
responsibility ([`2026-09-20-fan-epiplexity-diversity.md`](2026-09-20-fan-epiplexity-diversity.md),
[`2026-09-20-fan-select-then-commit.md`](2026-09-20-fan-select-then-commit.md)); teacher
forcing is a bypass and per-pass targets are met in one step
([`failures/2026-09-21-lxtul-loop-denoise.md`](../../../../lab/experiments/failures/2026-09-21-lxtul-loop-denoise.md));
the recipe rewards idempotence (the terminal fixed-point term asks the last pass to change
nothing; sampled depth trains the exit to look the same at any T), and turning the term off
is safe on the fan (fp0: rank unchanged, scale held elsewhere).

The problem tree, as engineering problems:

```
LXTUL works = channel worth it × passes have a job × reader cashes it × K are candidates × entry stable
├── A. Channel      A1 tokens bypass (forced share 0.19)   A2 bandwidth (0.21 of 0.40 lost)   A3 first-token lookup
├── B. Passes       B1 teacher forcing bypass   B2 idempotence rewarded   B3 no new input after pass 1   B4 relay never forced
├── C. Reader       C1 frozen coda is an OOD penalty   C2 selector before the span cannot choose
├── D. K streams    D1 scalars get gamed   D2 seed noise escapes by scale   D3 WTA makes candidate value
├── E. Target       E1 VAE codes are context-blind   E2 whole-span decoder + write-all moved the channel
└── F. Entry        scale growth, detonation (ramp + hinge)
```

## Proposal

### Step 0, the measurement that bounds the prize (one small build, no loop)

A **reach dial on the budget mask**: `model.span_reach: int` under `model.span_mask:
span`, the plain-model knob the 0.40 budget was measured with
([`2026-09-11-arc-span-budget.md`](../../../../lab/experiments/failures/2026-09-11-arc-span-budget.md);
no TUL in the model at all, so the endpoints are exact by construction). Relation
`causal AND span_id[j] >= span_id[i] − r` on both attention branches and the core; 0 is
bit-identical to `span`, −1 is fully causal; the conv / value-shift and bigram stay cut at
every boundary for every r, so `reachall` differs from `row` by those local routes alone
(a fourth number the existing `full` arm gives for free). Corrected 2026-09-21 21:28 from
an earlier draft that put the dial inside `tg_strict_allow`. Three arms at HEAD (`span`
re-run, `reach1`, `reachall`), 5k, 480 rows, paired; prereg
[`planned/2026-09-21-span-reach-split.md`](../../../../lab/experiments/planned/2026-09-21-span-reach-split.md).
Two numbers fall out:

- **bandwidth ceiling** = CE(r=0) − CE(r=1): what a perfect memory of the PREVIOUS span is
  worth. The slot channel returns 0.19; this says how much of the rest is history a
  compressed cell cannot carry.
- **far budget** = CE(r=1) − CE(r=−1): what spans further back are worth. This is the
  relay loop's entire job, and the ceiling of any honest reach K-curve.

Build size: one relation function (`span_reach_allow`), one config field, two YAMLs, and
tests in `tests/test_span_mask_leak.py` (reach 0 bit-identical, the hand-built matrices,
and the leak test with a perturbation one span back that must move and two spans back
that must not). The compose check prints the value at startup.

Reading rule, frozen now: if the far budget is under 0.03 nats, no geometry gives this
loop a depth job at this span size, and the reach arm is NOT queued; the design falls
back to the two-channel alternative below. If it is 0.10 or more, the reach arm is queued.
Between, Wolfe decides with the two numbers in hand.

### Step 1, LXTUL-R: fan4-all plus the chain (one factor from the write-all arm)

`morph/configs/tul_slot_spandec_strict_fan4_all_reach1.yaml`, defaults
`tul_slot_spandec_strict_fan4_all` + `_self_`:

| problem | knob (exists) | value | why this value |
|---|---|---|---|
| A1 / B3 / B4 forced relay | `tul.loop_reach` | 1 | span k−t reaches slot k only at pass t−1; depth is reach by construction |
| A1 (the coda side) | `tul.tg_coda_prefix_reach` | `prev` | the coda reads the previous slot's cells only, so nothing routes around the chain |
| C1 / C2 reader | live coda, `tul.fan_mix: all`, `tul.fan_all_wta_lambda: 1.0` | inherited | the per-token read of all K cells cashes what a selector could not |
| D1 / D3 candidates | `tul.fan_repel_mode: epivol`, `fan_k: 4`, `prefix_k: 4` | inherited | the one diversity term not gamed; WTA responsibility |
| B2 idempotence | `model.core_fixed_point_lambda` | 0.0 | the term asks the last pass to change nothing; fp0 showed 0 is safe on the fan |
| F scale | `model.slot_state_renorm` | true | the state keeps its entry norm per slot; the carry arm's unbounded RMS (0.5 → 75) is the failure this bounds. Measured only on E14 (`g102-rn`); on the reach chain it is a prediction |
| F detonation | `training.warmup` | **1000** (inherited from `base.yaml`, composes to 1000 through the whole strict/fan chain; asserted at startup) | the measured cure, 0 of 9 detonations; never override to 0 on the ternary + AdEMAMix recipe |
| F detonation | `model.slot_gain_lambda` / `slot_cot_clip` | 100 / 4.0 inherited | the slot-loop gain constraint |
| E target | `tul.spandec: true` | inherited | the whole-span decoder is the aux that moved the channel (+0.072) |
| B1 | no per-pass target, no teacher forcing | by omission | the loss is token CE through the live coda plus WTA on the rolled-out exits |
| depth | `model.mean_depth 6`, `max_depth 8`, sampled | inherited | Wolfe: sampled depth is the recipe; fixed depth learns a different thing |

Horizon: 5k for the K-curve, the per-hop bins and the planted decay row; 20k for the CE
pairing if 5k passes.

### Step 2, the decay fix, only if Step 1 reproduces the decay row

The hop-distance filing's own order: a persistent slot component written once on arrival
and not re-processed, then a per-pass write gate on the cross-cell read, then width. The
carry filing adds the constraint every candidate must meet: keep the direct pass-1 read
(add, never replace) and bound the state by construction (a running mean or an EMA with
a fixed target RMS). One candidate that needs no new mechanism: `tul.prefix_source:
trajectory` on the chain, so the coda reads cell t of slot k as the state AT ARRIVAL of
span k−t, before the later passes decay it. Its known artefact (forced-depth pad cells
inflating the K-curve, [`trajectory-prefix`](../../../../lab/experiments/failures/2026-09-13-arc-trajectory-prefix.md))
must be closed first: pad cells beyond the realised depth are masked in the coda's read,
and the K-curve is read on realised-depth rows only.

### What z is, stated so the objection is on the record

Under reach, z holds history as well as the next thought. Wolfe's objection stands and
this design accepts the cost for one reason: relay is the only operation the measured
evidence says a pass can have here. The alternative that keeps z as the plan alone is the
**two-channel design**: history through the coda's token reach (Step 0's dial left at r=1
or −1, no loop needed), plan through the K cells with WTA. It is cheaper and it does not
force depth; it is the fallback if Step 0 says the far budget is small.

## Alternatives considered

- **The end-to-end rollout denoiser (no teacher forcing).** Removes the bypass, but it is
  then a T-step regression onto a context-blind code from a noise entry; per-pass targets
  are met in one step and arm A read 0.126 better than the sample. Not queued.
- **More inside-the-loop levers.** Nine arms closed them (fp0, trig, noise, lineage, np0,
  denoise, tlow, vq, width). Not queued.
- **The paid loop (tokens through the core).** Rejected by Wolfe 2026-09-09: TUL means the
  slot loop; think once, decode cheap.
- **Fixed depth.** Wolfe 2026-09-14: sampled depth is superior and learns a different
  thing; a fixed rung is not ranked against it.
- **Re-supply carry (sum / gate).** Filed failure: replaced the read, diluted the far
  hops, unbounded state.
- **A wider single cell instead of K streams (pk16 / pk32).** Width is a reader lever
  (0.022 at 4 cells, 0.035 at 8) and not a depth lever; it is Step 2's third candidate, not
  the arm.
- **Longer horizon.** The 5k → 20k table above rules it out as the lever.

## Acceptance criteria

Step 0 (the dial), frozen before the run in its own planned file:

- A0.1 the dial composes and prints; the leak test passes with r=0 bit-identical to `span`.
- A0.2 CE(r=0) − CE(r=−1) reproduces the 0.40 [0.38, 0.42] budget within 0.05 on 480 rows.
- A0.3 the two numbers (bandwidth ceiling, far budget) are filed with intervals; the
  reading rule above decides Step 1 without a second look at the data.

Step 1 (LXTUL-R), frozen before the run in its own planned file; bars set now:

- A1.1 compose check at startup prints `warmup 1000`, `core_fixed_point_lambda 0.0`,
  `loop_reach 1`, `tg_coda_prefix_reach prev`, `fan_mix all`, `slot_state_renorm true`.
- A1.2 healthy: `preclip/total` under 1e4 at every step ≥ 200; entry norm growth under 3x
  over training (the noise arm's 11x is the failure).
- A1.3 token K1−K6 at 5k on 480 rows at or above **+0.05** (the single-cell chain gave
  +0.0163; the bar is three times that) AND at or above half of Step 0's far budget.
- A1.4 the planted decay row: a copy two spans back keeps at least **75 %** of its
  arrival worth at depth 6 (the ruler kept 24 %: 0.148 → 0.035).
- A1.5 paired CE at 5k vs `slot-spandec-strict` at or above **−0.02** (the chain at CE
  parity; the fan's write-all was +0.034 better than pk4, so a loss here is a defect).
- A1.6 the streams are candidates: `rank_t1` ≥ 2.5 of 3; `oracle − mixed` quoted beside
  the 0.04 near-copy floor.
- A1.7 rate at or above 0.80x fan4-all's tok/s.
- A1.8 (added 2026-09-21 21:28, the depth-dependence guard) paired CE at 5k vs Step 0's
  `budget-web-reachall` model, reported as the fraction of the far budget the loop at
  depth 6 recovers; the reach K-curve is forced by construction (the `loop_reach` comment
  says so), so a pass on A1.3 without this fraction is depth dependence, not a win.

Pass = A1.1 to A1.7 all hold. If A1.3 holds and A1.4 fails, Step 2 runs. If A1.3 fails
with A1.4 holding, the reach frame is closed for this loop and the two-channel design is
the next note.

## Risks

- **The object.** z as history is the cost the design pays for a depth job. If Step 0's
  far budget is small, the cost buys nothing and the two-channel design is the answer.
- **Blinding buys the curve.** A chain is a worse model than an unrestricted one at equal
  compute (`matched-compute plain beats the slot family`); the K-curve is honest reach, not
  a win over plain. The design scores on loop contribution by Wolfe's rule and says so.
- **Decay.** The per-pass decay of carried content is the map's own property; Step 1 may
  reproduce it with K streams, and Step 2 has three candidates, none measured on the chain.
- **Renorm on the chain is a prediction**, measured only on E14 arms.
- **Trajectory pads.** The Step 2 candidate carries a known artefact; it must be closed
  in the read before it is scored.
- **Compose before queue.** Every config here composes through Hydra and `tul_setup`
  before it is queued (`compose-every-config-before-queueing`); unknown `tul.*` keys raise.
- **One trainer on the 5090.** Step 0's three runs are 5k each; Step 1 is 5k then 20k.

## Outcome log

**Step 0 outcome (2026-09-22 02:05).** Filed under
[`failures/2026-09-21-span-reach-split.md`](../../../../lab/experiments/failures/2026-09-21-span-reach-split.md)
(P-4, P-8, P-9 fail; 6 of 9 hold). At 5k, depth 6, 481 paired blocks: span-h 4.3975,
reach1 4.2181, reachall 4.0922. The far budget's LOWER bound (reach1 − reachall) is
**+0.1259 [+0.1165, +0.1364]**, so the reading rule's "queue" clause fired; Step 1 is
planned in `lab/experiments/planned/2026-09-22-lxtul-r-step1.md` (bcb0dc2) and launched
2026-09-22 01:38. The previous-span value's UPPER bound (span-h − reach1) is +0.179. The
fourth arm (reach 1 at coda[0] only) reads WORSE than span-h by 0.044 (0.39 better on the
first token after a boundary, 0.106 worse from the eighth on); eval swaps on the Spark
show the mask is harmless to a model trained without it (−0.001) and the trained model
uses the read at every offset, so the deficit is on the trained weights; a learned
single-block trade and a run draw are not separated by one arm (the same-config
cross-commit pair drifts 0.048 at 8+). The far budget's honest bracket is [0.126, ≤ 0.28];
the relay share is not read. A0.1 to A0.3 hold on the lower bound. Observation: the plain
loop's own K1−K6 is largest under reach 1 (+0.055 vs +0.033 at reach 0 and +0.038 at reach
all): a relay geometry makes the plain loop's passes relay.

**Step 1 outcome (2026-09-22 04:39), a method fault.** Filed under
[`failures/2026-09-22-lxtul-r-step1.md`](../../../../lab/experiments/failures/2026-09-22-lxtul-r-step1.md).
The arm (`slot-spandec-strict-fan4-all-reach1`, bcb0dc2) ran healthy and read token K1−K6
**+0.0202 [+0.0193, +0.0212]** at 5k (bar +0.063; the largest slot-loop K-curve on the
ledger), paired −0.033 vs strict, +0.019 vs fan4-all-fp0 at depth 6 and +0.040 at depth 1
(A1.8 fraction −0.15), rank 2.98, 7,496 tok/s, entry norm 1.27x, channel worth 0.162 (fan4-all
0.196), planted g = 2 kept 89 % at depth 6. But the planted row read content three and four
spans back at depth 1 (0.048, 0.023; exact zeros were due), and a depth-1 perturbation test
on the tiny model confirmed it: the register branch passed `tg_relation` without `tg_seg`,
so the CCA conv and the value shift relayed about one slot per LAYER (four slots per pass on
the real model). The composition's A1 checks covered attention masks only. Fixed at 9b430d3
(per-slot `tg_seg` under `loop_reach > 0`, reach 0 untouched; exact zeros beyond one slot per
pass in `tests/test_lxtul_r_composition.py`). None of the Step 1 binding clauses is applied;
Step 1b (`lab/experiments/planned/2026-09-22-lxtul-r-step1b.md`, 0b338f4) re-runs the arm
at the corrected geometry, queued behind the seed twins of the Step 0 coda arm
(`2026-09-22-coda-seed-twin.md`, 599ae1d). Risk row to add for the next composition:
"every op with a receptive field, and which kwarg cuts it".

**Seed-twin correction to the Step 0 outcome (2026-09-22 05:00).**
[`failures/2026-09-22-coda-seed-twin.md`](../../../../lab/experiments/failures/2026-09-22-coda-seed-twin.md):
the seed-1 coda arm was a bad draw of the single-block-reach config (two seeds 0.152 apart;
the plain span config's seeds 0.004 apart). At seed 2 one non-looped block reading the
previous span is worth 0.112 at every offset. Step 0's brackets: previous span [0.112,
0.179], far budget [0.126, 0.197], relay share 0.071. The queue rule's reading is
unchanged (lower bound 0.126). Two consequences for this note: the two-channel fallback's
"history through the coda's token reach" is worth about 0.11 at ONE block and 0.18 at every
block, and any single-block design must be read at two seeds.

**Step 1b outcome (2026-09-22 09:51).** Filed under
[`failures/2026-09-22-lxtul-r-step1b.md`](../../../../lab/experiments/failures/2026-09-22-lxtul-r-step1b.md):
P-3 and P-4 fail, the rest hold (P-9, the leak check, holds on the real model: exact zeros
before pass g − 1). At the corrected one-slot-per-pass geometry the arm reads token K1−K6
**+0.0261 [+0.0251, +0.0273]** (bar +0.063; the largest slot-loop K-curve on the ledger,
K3−K6 +0.0057), paired −0.026 vs strict, **+0.0265 vs fan4-all-fp0 at depth 6 and +0.053
at depth 1** (A1.8 fraction −0.21: the geometry costs 0.053 and the passes recover half),
rank 2.97, 6,335 tok/s, entry norm 1.29x, tripwire spikes 1,050 and 431 (under 1e4, the
family's first), channel worth 0.153 (fan4-all 0.196), planted g = 2 kept 62 % (bar 75 %;
the probe is an exact-induction test on out-of-context subwords with a 15.5-nat control,
4.7 above uniform, and is filed with that caveat). Readable check on 48 rows: 4.267 nats,
ppl 71, top-1 29.5 % against fp0's 4.236 / 69 / 29.8 %; the checkpoint predicts text. The
binding's branch is "both fail: `failures/`, no Step 2, Wolfe decides" between Step 2 (a
persistent carry on this chain) and the two-channel design; this note's Step 1 is not
promoted.
