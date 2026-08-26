# Experiment: the span bag-mean pools slot-state diversity out as 1/sqrt(L)

Status: success

**Process note, stated rather than hidden.** The prediction below — slope −0.5 — was made
in conversation BEFORE the probe was written or run, and the arithmetic
`sqrt(32/11) = 1.71` against a previously measured 1.78x was the reason for running it. It
was not committed to `lab/experiments/planned/` first. The prediction is genuinely prior
to the data; the FILE is not. Read the slope as a confirmed prediction and the rest of this
document as description.

## Question

Reviewer's hypothesis, 2026-08-24: pooling the tokens of a span into one vector is what
hurts the looping backbone, in the forward signal and in credit assignment both. Raised
against a measurement showing `tul.span_cap` 32 -> 12 raising the core input effective rank
1.78x, with the objection that a cap should not matter much if it only trims a tail.

## The mechanism, before any measurement

Spec 3.2: a slot's input is `E_slot + mean_j embed(t_j)` over its span. A PLAIN mean. Two
consequences follow from that one line and neither needs a model to derive.

**Forward.** For roughly independent token embeddings the mean of L of them has covariance
`Sigma / L`, so its deviation from the corpus mean shrinks as `1 / sqrt(L)`. A slot input is
therefore a large CONSTANT — `E_slot`, shared by every slot — plus a signal that vanishes
with span length.

**Backward.** `d(bag_mean) / d(embed_j) = 1 / L` exactly. A token inside a 32-token span
receives one thirty-second of the credit a token in a 1-token span would. That is exact
arithmetic, not a measurement, and it is recorded here because it is the other half of the
hypothesis and needs no probe.

**Prediction.** `log ||v_i - vbar|| = a - 0.5 log L_i`. A slope near 0 kills the hypothesis;
a slope near −0.5 means every doubling of span length costs a factor 1.41 of slot-state
spread.

## Method

`lab/divergence/pooling_probe.py`, one forward per configuration on `ROLL_step_1700`
(healthy, core share 0.042), batch 6. `slot_input` is a plain method rather than `forward`,
so it is wrapped for the duration of the call and restored in `finally`. `span_len` is
populated only when the TUL gate is configured, so L is DERIVED from `bag_id` as the number
of token positions carrying each slot's id — the same count the bag-mean divides by.

Effective rank is the participation ratio of the singular values of the CENTRED slot
inputs, reported at a FIXED slot count as well as over all slots, because effective rank is
bounded by (rows − 1) and the configurations produce different numbers of slots.

## Results

**The law holds, three times, independently.**

| configuration | slots | span len mean / median / q90 | slope | r2 |
|---|---:|---|---:|---:|
| `span_cap` 32, `max_slots` 64 (shipped) | 342 | 18.04 / 17 / 32 | **−0.473** | 0.931 |
| `span_cap` 12, `max_slots` 128 | 621 | 10.29 / 12 / 12 | **−0.504** | 0.863 |
| `span_cap` 8, `max_slots` 160 | 850 | 7.45 / 8 / 8 | **−0.527** | 0.711 |

All three land within 0.03 of the predicted −0.5.

**And the deviation depends on L ALONE, not on the configuration.** Mean deviation within
matched span-length buckets, across the three runs:

| span len | cap32 | cap12 | cap8 |
|---|---:|---:|---:|
| 4–5 | 0.5073 | 0.5068 | 0.5002 |
| 6–7 | 0.4182 | 0.4126 | 0.4197 |
| 8–11 | 0.3405 | 0.3398 | 0.3674 |

That is the control the slope alone does not give: a configuration's entire effect on slot
diversity is mediated by the span-length distribution it produces. Nothing else about
changing the cap matters.

**What the cap buys**, at a fixed 200 slots so the ranks are comparable:

| configuration | eff rank at n=200 | mean deviation | deviation / \|\|E_slot\|\| |
|---|---:|---:|---:|
| `span_cap` 32 | 27.49 | 0.2961 | 0.9888 |
| `span_cap` 12 | 44.23 | 0.3430 | 1.1453 |
| `span_cap` 8 | **54.03** | 0.3883 | **1.2966** |

`span_cap` 32 -> 8 raises slot-input effective rank **1.97x** and the signal-to-constant
ratio **1.31x**.

## The premise correction

The objection assumed a median span near 11, which would make a cap of 12 a tail trim. The
measured median under the shipped rule is **17**, with mean 18.04, q75 = 28 and q90 = 32.
The cap at 32 already binds for more than 10 % of spans and a cap at 12 moves more than half
the mass. So the 1.78x was not a tail effect and needed no explaining away — but the
hypothesis it prompted is correct anyway, and now has a law behind it.

## What this does and does not settle

**Settled.** Pooling is a real, quantified, first-principles limit on slot-state diversity,
with a confirmed exponent and a clean control. Span length is the knob and it is
config-only.

**Not settled, and this is the important limit.** Pooling costs about 2x of slot-input
effective rank between the shipped setting and `span_cap` 8. The collapse this campaign is
chasing takes the slot states from an input rank near 28 down to a post-loop effective rank
of **1.7 to 4.8**. The loop destroys roughly 10x more diversity than the pooling does.
Pooling is a genuine contributing term; on these numbers it is not the dominant one, unless
the loop's contraction is a threshold effect that the input rank tips.

Whether raising the input rank propagates THROUGH the loop is a separate measurement and it
is the one that decides whether this is a fix or a footnote.

## Not verified

* One checkpoint, one batch, one boundary rule. The law is confirmed across three
  configurations but at a single point in training.
* The 1/L credit-assignment half is exact arithmetic and is NOT measured here.
* Nothing causal. No arm has been trained at a different `span_cap`.
* `span_cap` 8 and 12 need `max_slots` 160 and 128 to avoid the packer's early row end.
  Those fit at batch 6 (14.49 GB measured) but were never run at the batch 10 to 14 the
  earlier arms used.
