# Planned: the reach split — how much of the 0.40-nat cross-span budget is the PREVIOUS span, and how much is further back

Status: failure

Date: 2026-09-21 (frozen before any GPU step of the three arms). Arc: Step 0 of the
LXTUL-R design note
[`2026-09-21-lxtul-r-reach-composition.md`](../../../.agents/notes/proposed/architecture/2026-09-21-lxtul-r-reach-composition.md).
Ruler: the span-budget pair
[`failures/2026-09-11-arc-span-budget.md`](../failures/2026-09-11-arc-span-budget.md)
(`budget-web-full` 4.0394, `budget-web-span` 4.4389 at depth 6 on 480 rows, gap +0.3994
[+0.3838, +0.4162]; commit cc4e034). Wolfe 2026-09-21: "use a subagent to build, verify its
work, then run."

## Question

When nothing crosses a span boundary, web text loses 0.40 nats at 5k. The slot channel
returns 0.19 of that. Of the 0.40, how much is the previous span alone (what a perfect
one-span memory is worth) and how much is spans further back (what a relay loop would
have to fetch across passes)? The second number is the ceiling of any honest reach
K-curve and decides, by a rule frozen here, whether the LXTUL-R reach arm is queued.

## Hypothesis

The budget is front-loaded in TIME as well as in offset: the previous span carries the
larger part (the local topic, the open clause, the names just used), and spans further
back carry a smaller but real part (recurring names and copied strings, which need
induction over long range). Against it: web text at seq 1024 has 30 to 60 spans per row,
and the offset-8+ gap of 0.31 nats (tokens that already have 8 to 31 tokens of same-span
context) says the loss is not only local, so the further-back part could be the larger.

## Method

One knob on the plain model, no TUL: `model.span_reach: int` under `model.span_mask:
span` (build: `morph/model/tul_layout.py::span_reach_allow`, threaded through
`transformer.py` and `build_morph_config`; tests in `tests/test_span_mask_leak.py`). The
relation is `causal AND span_id[j] >= span_id[i] − r`; `r = 0` is bit-identical to
`span`, `r = −1` is fully causal attention. The conv / value-shift and the bigram stay
cut at every boundary for every r, exactly as under `span`, so `reachall` differs from
`full` (`row`) by those three local routes alone. That difference is a fourth number the
existing `full` arm gives for free.

Three arms at HEAD, same seed, same recipe as the budget pair (`budget_root.yaml`:
`notul_panel_norm_match`, seq 1024, batch 6, 5,000 steps, `training.warmup` 1000,
`norm_match`, prune / carve / route off), differing in `model.span_reach` and
`wandb.name` and nothing else (the compose diff is printed before queueing):

| arm | config | `span_reach` | attention reach |
|---|---|---|---|
| `budget-web-span-h` | `budget_web_span` | 0 | own span only (the 2026-09-11 arm re-run at HEAD) |
| `budget-web-reach1` | `budget_web_reach1` | 1 | own span + the previous span |
| `budget-web-reachall` | `budget_web_reachall` | −1 | every earlier span (conv / bigram still cut) |
| `budget-web-reach1-coda` | `budget_web_reach1_coda` | 1 at the coda's first block only | own span + the previous span's reach-0 states, no relay (added 2026-09-21 21:46) |

The `span` endpoint is re-run at HEAD rather than reused so the three arms share one
commit; the 2026-09-11 arm's sweep at cc4e034 is paired against the re-run as a drift
check. Runner line kind `plain`, sweeps at 2500 and 5000, 480 rows; readouts are the
runner's forced-depth sweep and `paired_vs_ruler.py` on `tok_index`.

Readings (all at 5k, depth 6, paired on the 480 rows):

- **bandwidth ceiling** = CE(span-h) − CE(reach1): what a perfect memory of the previous
  span is worth.
- **far budget** = CE(reach1) − CE(reachall): what spans further back are worth through
  attention.
- **local routes** = CE(reachall) − CE(full, 2026-09-11 at cc4e034): the conv / value-shift
  / bigram routes' share, read across commits (a drift caveat is attached to it).
- the per-offset table of the budget filing, for each pair.

**Method amendment 2026-09-21 21:46 (before any GPU step; the builder's finding).** The reach is
applied at EVERY attention layer, so the stack relays content one span per layer: a
token in span k reads span k−1's states, which after the first layer already hold span
k−2's content. Measured on the leak test's fixture: a perturbation two spans back moves
end-to-end logits by about 0.87 under `span_reach: 1` while the single-layer relation is
exactly zero there (`tests/test_span_mask_leak.py::test_reach1_relation_is_reach1_at_a_single_attention_layer`).
`budget-web-reach1` is therefore NOT "the previous span alone". It bounds the split
from one side. A fourth arm gives the other side: `budget-web-reach1-coda`
(`budget_web_reach1_coda.yaml`, `model.span_reach_layer` = the coda's first block, a
build of 2026-09-21): the reach-1 relation at ONE non-looped block and reach 0
everywhere else, so a query reads the previous span's states as computed under reach 0
and no relay exists (an end-to-end perturbation two spans back moves logits by exactly
0, which is that arm's leak test). The readings become brackets:

- previous-span value: lower bound CE(span-h) − CE(reach1-coda) (one block reads it),
  upper bound CE(span-h) − CE(reach1) (every layer reads it, plus relay);
- far budget: lower bound CE(reach1) − CE(reachall), upper bound
  CE(reach1-coda) − CE(reachall);
- relay share = upper − lower of the far budget (what the per-layer stack fetched from
  beyond one span by relay).

Ordering the arms have to respect: CE(span-h) ≥ CE(reach1-coda) ≥ CE(reach1) ≥
CE(reachall), each step within its interval. Four arms, about 3.3 hours.

## Predictions (frozen)

- **P-1 (survival).** All three arms HEALTHY to 5,000 (`preclip/total` under 1e4 at every
  step ≥ 200). **90 %.** Same recipe as the pair that held; the ramp is on.
- **P-2 (the endpoint reproduces).** CE(span-h) − CE(full at cc4e034) inside
  [0.34, 0.46] on the 480 rows. **80 %.** The drift check: if this fails, the split is
  read but every number carries a "code drift since cc4e034" caveat and the `full`
  endpoint is re-queued at HEAD before the design's reading rule is applied.
- **P-3 (the sum closes).** bandwidth ceiling + far budget + local routes = the budget,
  within 0.03 of CE(span-h) − CE(full). **85 %.** An identity up to the cross-commit term;
  it is listed so an arm that silently differs by more than its knob is caught.
- **P-4 (the previous span is the larger part).** bandwidth ceiling above **0.20** nats.
  **65 %.** Point estimate 0.24. The offset-1 gap (0.96) and offset-2 gap (0.69) are the
  clause the previous span opened.
- **P-5 (the far budget, the headline).** far budget inside **[0.06, 0.20]**: **55 %**.
  Point estimate 0.12. Above 0.20: **20 %** (induction over long range is most of the
  offset-8+ loss). Below 0.06: **25 %** (one span of context saturates what a 270M model
  uses at 5k).
- **P-6 (the local routes are small).** local routes below **0.06** nats. **70 %.** The
  conv reaches a few tokens and the bigram one; attention to the same tokens covers them.
- **P-7 (the far budget is at every offset).** In the reach1 − reachall per-offset table,
  the 8+ bin's gap is at least **half** the offset-0 bin's gap. **60 %.** Long-range
  context is topical and lifts every position; the previous-span part is what is
  front-loaded.

**Scoring note and two predictions added 2026-09-21 21:46, before any GPU step (the fourth arm).**
P-4 is scored on the previous-span value's UPPER bound (span-h − reach1), P-5 on the far
budget's LOWER bound (reach1 − reachall): the design's "queue" case needs the lower bound
and its "stop" case needs the upper bound, so both are read and the binding below says
which one each clause uses.

- **P-8 (the relay share).** far-budget upper − lower inside **[0.02, 0.12]**. **60 %.**
  Point estimate 0.06. The per-layer stack has 14 attention layers and the hop-distance
  filing measured relay decaying a third per hop inside the loop; a plain stack should
  relay better than that but not fetch everything. Above 0.12: 25 % (relay does most of
  the far work, and then a loop that relays one span per pass has the same job). Below
  0.02: 15 %.
- **P-9 (the ordering).** CE(span-h) ≥ CE(reach1-coda) ≥ CE(reach1) ≥ CE(reachall) with
  each consecutive gap at or above −0.01 (a violation larger than that is a build or
  seed fault, not a reading). **85 %.**

## Binding (the design note's rule, copied so it cannot drift)

- far budget UPPER bound **under 0.03**: no geometry gives the slot loop a depth job at
  this span size; the LXTUL-R reach arm is NOT queued; the next note is the two-channel
  design (history through the coda's token reach, plan through the K cells).
- far budget LOWER bound **0.10 or more**: the LXTUL-R arm (Step 1) is queued with its
  own planned file; its K1−K6 bar is +0.05 AND half of the far budget's lower bound, and
  its CE is paired against `budget-web-reachall` as well as against strict.
  (Bounds wording added 2026-09-21 21:46; the thresholds are the design note's, unchanged.)
- **between 0.03 and 0.10**: Wolfe decides with the two numbers in hand; nothing is
  queued by default.
- P-2 fails: re-queue `full` at HEAD before applying the rule.

## Not verified before launch

- No GPU step of the new relation; the runner's 12-step smoke is the first. The tests are
  CPU, eager, small shapes.
- Whether `span_ids_from_ids` numbers spans monotonically per row (the builder is asked to
  quote it; the relation is written to hold either way).
- The conv kernel width (the builder is asked to quote it), which bounds how far the
  still-cut local routes could have reached.
- Wall clock: the 2026-09-11 arms took about 49 minutes each; three arms is about 2.5
  hours on the 5090, one trainer at a time.

## Results

Four arms on the runner, 2026-09-21 21:59 to 2026-09-22 01:35 (the first launch at
438cf93 died at the first eval on a six-day-old plain-path regression, fixed at 66fc75c
with `tests/test_train_evaluate_plain.py`; the four arms ran at 66fc75c). Readouts: the
runner's forced-depth sweeps at 2500 and 5000, `paired_vs_ruler.py` on `tok_index`
(1024-token blocks, 481 blocks), `span_budget_profile.py` per offset. Artifacts:
`../results/2026-09-21-span-reach/` (paired jsons, profile json/txt, the runner's sweep,
anatomy and init-probe jsons). Intervals are the block bootstrap.

| arm | verdict (`preclip/total` max @ step) | final val | CE d1 | CE d6 | plain-loop K1−K6 |
|---|---|---|---|---|---|
| `budget-web-span-h` | HEALTHY, 14.1 @ 1284 | 4.4168 | 4.4300 | 4.3975 | +0.0325 |
| `budget-web-reach1` | HEALTHY, 11.2 @ 471 | 4.2347 | 4.2735 | 4.2181 | +0.0554 |
| `budget-web-reachall` | HEALTHY, 118 @ 261 | 4.1106 | 4.1300 | 4.0922 | +0.0378 |
| `budget-web-reach1-coda` | HEALTHY, 38.3 @ 2048 | 4.4539 | 4.4680 | 4.4414 | +0.0265 |

Paired depth-6 readings (arm A − arm B, positive = A worse):

| reading | value |
|---|---|
| span-h − full @ cc4e034 (the endpoint, P-2) | **+0.3581 [+0.3437, +0.3746]** |
| span-h − span @ cc4e034 (drift since 2026-09-11) | −0.0414 [−0.0435, −0.0393] |
| span-h − reach1 (previous span, UPPER bound, P-4) | **+0.1794 [+0.1742, +0.1850]** |
| reach1 − reachall (far budget, LOWER bound, P-5) | **+0.1259 [+0.1165, +0.1364]** |
| span-h − reachall | +0.3053 [+0.2944, +0.3177] |
| reachall − full @ cc4e034 (local routes, P-6) | +0.0528 [+0.0479, +0.0581] |
| reach1-coda − reachall (far budget, "UPPER" bound) | +0.3492 [+0.3392, +0.3604] |
| span-h − reach1-coda (previous span, "LOWER" bound; ordering, P-9) | **−0.0439 [−0.0485, −0.0393]** |
| reach1 − reach1-coda | −0.2233 [−0.2274, −0.2195] |

Per-offset gaps at depth 6 (`span_budget_profile_*.txt`, offset = tokens since the span
start; n per bin 21k to 24k, 304k in 8+):

| pair | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8+ |
|---|---|---|---|---|---|---|---|---|---|
| span-h − reach1 (previous span) | +0.125 | +0.682 | +0.394 | +0.287 | +0.246 | +0.221 | +0.179 | +0.154 | +0.111 |
| reach1 − reachall (far) | +0.087 | +0.125 | +0.197 | +0.197 | +0.161 | +0.138 | +0.134 | +0.126 | +0.114 |
| reach1-coda − reachall | +0.214 | +0.416 | +0.470 | +0.444 | +0.401 | +0.375 | +0.358 | +0.348 | +0.348 |
| reach1-coda − span-h (negative = coda better) | +0.003 | −0.392 | −0.121 | −0.039 | −0.005 | +0.016 | +0.046 | +0.068 | **+0.106** |

At depth 1 the coda row has the same shape (−0.386 at offset 1, +0.096 at 8+).

Scoring:

- **P-1 HOLDS (4 of 4).** Every arm HEALTHY; the largest preclip ratio 118 at step 261 on
  reachall, under the 1e4 tripwire.
- **P-2 HOLDS.** +0.358 [+0.344, +0.375] inside [0.34, 0.46]. The drift term is −0.041: the
  span endpoint reads 0.041 BETTER at 66fc75c than at cc4e034 on the same rows, so every
  cross-commit number below carries that caveat.
- **P-3 HOLDS.** 0.1794 + 0.1259 + 0.0528 = 0.3581, the endpoint gap to four places.
- **P-4 FAILS.** The previous-span value's upper bound is 0.179, under the 0.20 bar (point
  estimate was 0.24). Its lower bound from the fourth arm is not readable (P-9).
- **P-5 HOLDS (the headline).** The far budget's lower bound is 0.126 [0.117, 0.136],
  inside [0.06, 0.20]. **The design note's queue rule fired (0.10).**
- **P-6 HOLDS on the letter.** Local routes 0.053, under 0.06, but the interval sits 0.012
  from the bar and the cross-commit drift (0.041) is of the same size; the reading is
  "small", not "0.053".
- **P-7 HOLDS.** In the reach1 − reachall row the 8+ bin (0.114) is above half the offset-0
  bin (0.087); the far budget peaks at offsets 2 and 3 (0.197) and declines slowly from
  0.161 at offset 4 to 0.114 at 8+.
- **P-8 FAILS, unreadable.** Upper − lower = 0.349 − 0.126 = 0.223, above 0.12, but the
  upper bound is the coda arm's number and that arm fails P-9, so the relay share is not
  read from it.
- **P-9 FAILS.** CE(span-h) − CE(reach1-coda) = −0.044 [−0.049, −0.039], below the −0.01
  clause. The arm that reads the previous span at ONE block is worse than the arm that
  reads it nowhere. Per offset it is 0.39 nats BETTER on the first token after the boundary
  and 0.12 better on the second, then 0.106 WORSE from offset 8 on, where a token already
  has 8 to 31 tokens of its own span. The prereg pre-committed a violation of this size as
  a build or seed fault, not a reading.

Two "not verified" items from the prereg, now quoted: `span_ids_from_ids` writes a running
`np.cumsum` of boundary marks, so span ids are non-decreasing per row (the contract
`span_reach_allow` relies on, `morph/model/tul_layout.py`); the CCA conv kernel is 4
(`MORPHConfig.conv_kernel`, `morph/model/transformer.py`), so the still-cut local routes
reach at most 3 tokens back.

An observation with no clause: the plain loop's own K1−K6 is largest on `reach1` (+0.055
against +0.033 span-h, +0.038 reachall, +0.027 coda). Under per-layer reach 1 the six
looped blocks relay one span per layer, so more passes reach more spans; this is the
plain model doing what the design asks the slot loop to do, and it is the first plain
arm on the ledger whose loop earns more under a geometry change.

### The P-9 check: eval swaps on the DGX Spark (2026-09-22 01:40 to 02:00)

The prereg names a P-9 violation a build or seed fault. Four depth sweeps on the Spark
(worktree `MORPH-0922` at bcb0dc2, 480 rows, batch 3, depths 6 and 1; the Spark streams the
same rows: coda native 4.4415 there against 4.4414 on the 5090), each checkpoint under
both configs (`sweep_*_5000.spark.json`, `paired_swap_*.json`,
`span_budget_profile_swap_*`):

| model | config at eval | CE d6 | paired vs its native |
|---|---|---|---|
| reach1-coda 5k | `budget_web_reach1_coda` (native) | 4.4415 | — |
| reach1-coda 5k | `budget_web_span` (the read removed) | 4.5495 | +0.1080 [+0.1047, +0.1115] |
| span-h 5k | `budget_web_span` (native) | 4.3974 | — |
| span-h 5k | `budget_web_reach1_coda` (the read added, untrained) | 4.3963 | −0.0011 [−0.0013, −0.0009] |

Per offset (depth 6): removing the read from the coda model costs +0.057 at offset 0,
**+0.607** at 1, +0.287 at 2, +0.211 at 3, +0.162 at 4, +0.140 at 5, +0.113 at 6, +0.094 at
7 and **+0.043** at 8+. Adding the untrained read to the span-h model changes every bin by
0.001 to 0.004 in its favour.

What this settles. The block-10 relation is not a broken forward: a model that never saw
it loses nothing when it is switched on (−0.001). The coda model USES the read at every
offset, including deep in the span, and is still 0.106 worse than span-h at 8+. The
training path is the eval path: static graphs are off (`MORPH_STATIC_GRAPHS` unset, 0 log
mentions), `training.compile` covers the MLPs only (`compile_blocks` and
`compile_attention` false), and coda activation checkpointing runs only on the fan's
winner replay. So the arm trained on the relation it was scored on.

What one arm cannot separate. Either the single-block read is a learned trade (the coda's
first block spends its attention on the previous span, which pays at offsets 1 to 3 and
costs the own-span long-range read that the two remaining coda blocks cannot recover), or
the arm is a run-to-run draw. The calibration for the second: the same-config,
same-seed pair across commits (span-h at 66fc75c vs span at cc4e034,
`span_budget_profile_drift_*`) differs by +0.041 whole-arm and **+0.048 at 8+** with
+0.008 at offset 1; the coda arm's 8+ deficit is twice that floor and its offset-1 gain
is fifty times it. A twin (seed 2 of the coda arm, or the reach block moved to coda[1])
would separate the two; it is not queued, because neither reading changes the far
budget's LOWER bound or the rule it fired.

What the "upper bound" becomes. CE(reach1-coda) − CE(reachall) = 0.349 is not an upper
bound of the far budget: it contains the coda arm's own 8+ deficit. The loose honest
bracket is far budget in **[0.126, ≤ 0.28]**: the upper end is span-h − reachall (0.305)
less the previous-span value the coda arm proves at offsets 1 to 3 alone (about 0.028
whole-arm: 0.392, 0.121 and 0.039 over three bins of 24.5k tokens in 491.5k). The relay
share (P-8) is not read.

## Verdict

**Failure** (P-4, P-8 and P-9 fail; 6 of 9 hold). The headline holds: the far budget's
lower bound is **0.126 [0.117, 0.136]** nats at depth 6, the design note's queue rule
(0.10) fired, and Step 1 (`2026-09-22-lxtul-r-step1.md`, bcb0dc2) was queued and launched
at 01:38 on that bound alone. The previous-span value's upper bound is 0.179 (P-4's bar was
0.20): the previous span and the spans beyond it are closer to equal shares of the 0.40
than the hypothesis said. The fourth arm did not give the other side of either bracket:
one block reading the previous span makes an arm that is better on the first three tokens
after a boundary by 0.39 / 0.12 / 0.04 and worse from the eighth token on by 0.106, and
this filing's swaps show that is a property of the trained model, not of the mask.
Inconclusive on the brackets' other sides, so filed under `failures/`; what the method
could not distinguish is a learned single-block trade from a run draw, and the next
planned experiment on this question is a twin of the coda arm, held until Step 1 reads.

## Updated hypothesis

Of the 0.40 nats web text loses when nothing crosses a span boundary at 5k, at least
0.126 is beyond the previous span (reach 1 at every layer, relay included, still leaves
that much on the table against full reach), and at most 0.179 is the previous span read
through the whole stack. The far part is at every offset (0.087 at the first token,
0.197 at the second and third, 0.114 from the eighth on), so it is topical, not only
induction on the boundary. The plain looped core earns MORE depth under reach 1 (K1−K6
+0.055) than under reach 0 (+0.033) or reach all (+0.038): when the geometry routes far
content through a relay, the plain loop's passes do the relaying. That is the mechanism
Step 1 asks the slot loop to perform, and the reason the far budget is the right prize
for it. A single non-looped block reading the previous span is a different object: it
pays on the boundary tokens and costs the span's tail, and the cost is on the trained
weights, not on the mask.

## Correction (2026-09-22 05:00, after the seed twins)

`failures/2026-09-22-coda-seed-twin.md` (599ae1d): at seed 2 the single-block arm
(`budget-web-reach1-coda-s2`) reads **0.112 [0.108, 0.117] BETTER** than its span twin at
depth 6, at every offset (−0.076 at 0, −0.469 at 1, −0.069 at 8+), while the two span seeds
differ by 0.004 [0.001, 0.006]. The seed-1 coda arm read above was a bad training draw of
that config (its two seeds differ by 0.152), not a learned trade; the "learned trade" and
"run draw" readings in the swap section are settled in favour of the draw, and the
"single-block read costs the span's tail" sentence in Updated hypothesis is withdrawn.
Brackets re-read with the coda at seed 2 (the plain seed floor 0.004 is the error bar):
previous-span value **[0.112, 0.179]**; far budget **[0.126, 0.197]** (coda-s2 − reachall
+0.197 [+0.187, +0.208]); relay share **0.071**, inside P-8's [0.02, 0.12]; ordering
span-s2 4.4014 ≥ coda-s2 4.2893 ≥ reach1 4.2181 ≥ reachall 4.0922 (P-9's clause holds on the
seed-2 arm); P-3's sum 0.112 + 0.197 + 0.053 = 0.362 against the 0.358 endpoint. The
verdict's letter is unchanged (P-4, P-8, P-9 failed on the arms as run); the open item is
now the 0.041 drift between `span` at cc4e034 and `span-h` at 66fc75c, which the seed
floor shows is not noise.
