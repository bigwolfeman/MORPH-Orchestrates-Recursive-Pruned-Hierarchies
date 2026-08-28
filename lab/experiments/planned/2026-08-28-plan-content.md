# Planned: is the TUL plan EMPTY, or FULL BUT UNREAD?

Status: planned
Probe: `lab/divergence/plan_content_probe.py`   Spec: TG-WORKLIST.md A3
Written and committed BEFORE the probe was run on any real checkpoint.

## Question

Every plan measurement in this campaign reads z THROUGH the coda (plan-off ablations),
which cannot separate "z has no content" from "z has content the coda ignores". This
probe reads z directly with a throwaway decoder on frozen weights, and asks a second
question for free: is z a SUMMARY of its own span, or a PLAN for the next one?

## Hypothesis

Under `tg_restrict` the slot attends its ENTIRE own span through all four prelude
blocks, so z necessarily absorbs span i. In TG2 `emit_weight=0`, so the slot has NO
direct next-span supervision at all — its content is shaped only by being useful to the
coda. Therefore z should read much more like a summary of span i than a plan for span
i+1, and the plan's thinness is a TARGET problem, not a capacity problem.

## Predictions (frozen — do not edit during the run)

Numbers are nats/token on held-out batches. The deciding quantity is
`SHUFFLED - PLAN` ("nats saved by the real z"); the summary quantity is
`PLAN - SUMMARY`. Both are positive when the named effect is present.

- **P1 (not empty):** both TG2 seeds show `SHUFFLED - PLAN >= 0.05`. z is not inert.
- **P2 (it is a summary):** both TG2 seeds show `PLAN - SUMMARY >= 0.20` — z
  reconstructs its OWN span substantially better than the next one.
- **P3 (the restriction put it there):** TG2's `SHUFFLED - PLAN` exceeds the
  unrestricted control's (`ctrlworth-s3`) by `>= 0.05`.
- **P4 (readable at all):** the two TG2 seeds agree within `0.10`. If they do not, the
  n=1 rule applies and NOTHING below is readable.

I am NOT predicting whether `SHUFFLED - PLAN` clears 0.20 (the FULL band). That is the
number this probe exists to find out, and guessing it would be fitting the threshold to
a hunch.

## Decision rule

- P4 fails -> void the panel, add seeds, change nothing else.
- `SHUFFLED - PLAN >= 0.20` (FULL) -> the plan has content the coda is not reading. Next
  work is the READER (injection, per-layer projections), and the flow-matching TRAINING
  loss becomes well-motivated because there is content to sharpen.
- `SHUFFLED - PLAN <= 0.05` (EMPTY) -> next work is PROVENANCE and the TARGET. TG4a/TG4b
  are already queued for provenance; the target fix is a next-span objective.
- P2 holds (whatever P1 does) -> headline finding: z is a summary, not a plan. The TUL
  thesis has a TARGET problem, and any next-span objective (flow matching on span i+1,
  continuous MTP) is attacking the right thing.
- P3 fails -> the restriction is not what put content into z, which weakens the whole TG
  line independently of the other numbers.

## Method

Frozen `requires_grad_(False)` + `.eval()`; asserted, and a parameter fingerprint is
compared before/after the decoder fit so a leak through the tied `lm_weight()` cannot
pass silently. z is the output of `prefix_project` — what the coda actually reads — mean
over HC streams, concat over `prefix_k`. Blind non-autoregressive decoder, own fresh
unembedding, identical size/steps/LR/seed across all four conditions, fit and eval
batches disjoint (asserted). SHUFFLED permutes z across rows. Offsets past a target
span's true length are masked and counted; each row's last valid slot is excluded in
EVERY condition so N is identical. Corpus-unigram floor reported alongside.

Checkpoints, all at matched step 3000:
`tg2-s1`, `tg2-s2` (config `tul_tg2`), `ctrlworth-s3` (config `tul_a1`, unrestricted).

Self-test (`tests/test_plan_content_probe.py`, 16 tests): a synthetic positive control
where z encodes the target must show a large saving, and a noise-z negative control must
show ~0. A probe that cannot separate those two cannot be trusted here.
