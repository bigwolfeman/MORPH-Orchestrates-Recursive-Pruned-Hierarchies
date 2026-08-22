# Arm CW — force the latent to be load-bearing by deleting the cheap channel

Status: IMPLEMENTED, EVAL SCREEN RUN 2026-08-18. Not trained.

**Eval screen result** (`checkpoints/morph/tul-a1-acap1/step_20000.pt`, cut=576,
500 val batches, 3.52M scored tokens — full numbers and paired stats in
`ignore/Ai-notes/08-18-2026/arm-CW/RESULTS.md`):

| arm | CE (nats) |
|---|---|
| CW0 | 3.2985 |
| CW1 | 3.3439 |
| CW2 | 3.3530 |
| CW3 | 3.3732 |

CW0 < CW1 < CW2 < CW3, exactly the order predicted below. CW1 beats CW2 by 0.009
nats (95% bootstrap CI [0.0078, 0.0102], excludes 0, 500 paired batches) — CW2 does
NOT match CW1. Per this spec's own decision rule (below), that is the POSITIVE
result: the slot already carries something beyond a surviving position, unforced.
Small effect (~0.27% of CW0's CE), and this is still a SCREEN, not the training
run this spec's own motivation was building toward — see "Scope of this task".

## Why this arm and not another

Three measurements from 2026-08-18, all on `tul-a1-acap1/step_20000.pt`:

* `val/plan_nats = +0.0270` — removing the slots from the coda costs almost nothing.
* `val/first_tok_counterfactual = −0.1196` — the previous TOKEN beats the slot at
  predicting a span's first token, the one job the plan exists for.
* Slot-collapse probe — slot similarity tracks the next span at Spearman +0.2321,
  against +0.2187 for the raw bag-mean of the slot's own span. Six blocks of core
  loop add nothing measurable.

Read together: **the slot is approximately a bag-of-words of the span it closed.**
That is not a defect, it is the objective getting what it asked for. Token CE is
fully satisfied by a bag-of-words *while the tokens are still in the sequence*,
and gradient always takes the cheaper channel.

`tul.token_state_dropout = 0.15` is the existing acknowledgement of this — spec
line 534, "tax the cheap channel or the latent is ignored". It is a probabilistic
tax on a structural problem.

**This arm removes the cheap channel instead of taxing it.** Old token positions
are deleted from the coda's sequence; the slots remain. Predicting a late token
then has no path to early content except through a slot. If the latent still
carries nothing under that regime, the problem is not the objective and the
JEPA-style follow-up is not worth building.

## The change

One new config knob, `tul.coda_token_cut` (int, default 0 = off, bit-identical).

At the coda, when `coda_token_cut = C > 0`:

* **DROP** token positions with row index `< C`.
* **KEEP** every slot position, at every index.
* **KEEP** token positions `>= C`.
* Score CE on the kept token positions only.

Implement as a GATHER, not an attention mask. Spec §7.2 already records that the
fused kernels may not support a per-position mask, which is why `plan_nats` is a
gather. `_tul_coda_without_slots` (transformer.py:1510) is the template and does
the mirror operation — it keeps tokens and drops slots. Reuse `compact_index`,
`gather_positions`, and its `_g` padding helper rather than writing new indexing.

This is a GLOBAL drop, not a sliding window. Every query in the coda sees the
same reduced sequence. That is deliberate: a per-query window needs a mask, and
a global drop is both implementable today and a cleaner statement of the claim
("the coda sees the last L−C tokens plus every slot").

## The arms — the control is not optional

All four score CE over the SAME token positions (`>= C`) so the numbers compare.

| arm | early tokens | slots | what it settles |
|---|---|---|---|
| `CW0` | kept | kept | ceiling — the model with everything |
| `CW1` | dropped | kept | **the claim**: slots carry the early content |
| `CW2` | dropped, but an equal-KV-budget RANDOM subset retained | dropped | **the decider** |
| `CW3` | dropped | dropped | floor — nothing carries it |

`CW2` is the arm that makes this readable. `prefix_k=2 × max_slots=64 = 128`
slot positions, so CW2 retains 128 randomly chosen token positions from `[0, C)`
and no slots. **If CW2 matches CW1, the slot is not a summary — it is merely a
surviving position, and the whole latent-memory line is answered negatively.**
Seed the choice and log it. This is the same discipline that caught AGCLR being
three constants.

Expected ordering if the claim holds: `CW0 >= CW1 > CW2 > CW3`.
Any other ordering is a result and gets written down as one, not explained away.

## Scope of this task

1. Implement the mechanism and the four arms.
2. Tests, mutation-checked.
3. **Run the EVAL screen only** — all four arms on the existing
   `checkpoints/morph/tul-a1-acap1/step_20000.pt`, no training. This asks whether
   the slot ALREADY carries anything, and it is hours not days.
4. **Do not launch a training run.** One GPU job at a time on this machine and
   training is the operator's call.

The eval screen is a screen, not the experiment: the checkpoint was trained with
the tokens present, so a null here is consistent with "the latent was never
forced to carry anything" and does NOT by itself kill the design. Say that in the
report. A POSITIVE result here is the strong outcome — it would mean the slot
carries early content even though nothing made it.
