# Planned: the reach split — how much of the 0.40-nat cross-span budget is the PREVIOUS span, and how much is further back

Status: planned

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
