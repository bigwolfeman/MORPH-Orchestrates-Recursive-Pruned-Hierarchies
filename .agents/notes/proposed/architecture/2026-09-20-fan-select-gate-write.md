# Agent Note: the select arm writes the gate's own pick — close the train/eval mismatch before judging the selector

Status: proposed

## Problem

The select arm (`2026-09-20-fan-select-then-commit.md`, filed as a failure in
`lab/experiments/failures/2026-09-20-lxtul-fan4-select.md`) trained its coda on the
stream the TABLE chose: K no-grad passes read the span's target and picked the per-slot
winner, and that winner was written in 95 % of slots. At eval the gate's guess was
written. So the coda was trained on one input distribution and deployed on another. Its
deployed write sat 0.086 nats behind the width partner, the gate agreed with the table
33 % of the time, and the filing first read that as "a gate before the span cannot
predict the mode". Wolfe's reading (2026-09-20): the selection that reads the target at
train and not at eval is the fitted-z trap again, it cannot transfer to inference, and
the 5k gap is not a ranking. Nothing in the filed arm separates a gate that cannot
predict from a coda that never saw what the gate picks.

## Proposal

`tul.fan_select_write: oracle | gate | anneal` (`morph/model/tul.py`, read only under
`fan_mix: select`; `morph/training/tul_setup.py` keys, manifest, banner).

- `oracle` is the filed arm, bit for bit: the training pass writes the table's winner.
- `gate` writes the gate's own argmax on every valid, non-forced slot. The table still
  runs each step, as the gate's LABEL only (`select_gate_loss` on the table's winner).
  The forward is then the same at train and eval.
- `anneal` is per-slot scheduled sampling: a slot writes the gate's pick with probability
  `step / fan_select_write_anneal` (clamped at 1) and the table's winner otherwise. The
  trainer fills the non-persistent `fan_select_step` buffer each step (the same pattern
  as `code_grade_step`); it is read only inside the compile-disabled `_tul_fan_select`.
- The eps random write applies under every mode (`select_winners` runs first, a forced
  slot keeps its random stream). `MORPHTransformer._fan_select_written` is the one home
  of the choice.

Instruments added to the train-side stats: `select_write_p_gate`, `select_write_from_gate`,
`select_written_ce` (the table's CE of the stream actually written; minus
`select_oracle_ce` is the TRAIN-side selector regret, the partner of the eval-side
`mixed_ce − oracle_ce`), `select_written_agree`, `select_written_share_k{i}` (the collapse
instrument of a closed loop; `select_share_k{i}` stays the table's winner shares).

The queued arm is `tul_slot_spandec_strict_fan4_select_gate.yaml`: the select config with
`fan_select_write: anneal` over 1500 steps, one factor. Prereg
`lab/experiments/planned/2026-09-20-lxtul-fan4-select-gate.md`.

## Alternatives considered

- **`gate` from step 0, no anneal.** The gate is zero-init, so every slot would write
  stream 0 from the first step and the other three streams would never be read; the eps
  0.05 random write alone is a slow explorer. The anneal lets the table drive the coda's
  reads while the gate learns its label. Not queued first; `gate` exists as a mode for a
  later factor.
- **A clean argmin as the gate's label.** The filed arm trained the gate on the table's
  winner including the eps random picks (5 % label noise). Kept unchanged so the queued
  arm moves one factor; a later arm can vary it.
- **Straight-through or Gumbel softmax through the gate** (the gate's pick differentiable
  into the streams). Changes what the streams learn as well as what the coda reads; two
  factors. Not this arm.
- **Write-all (`fan_mix: all`)**, every stream written into its own prefix cell and the
  coda's per-token attention as the selector (`2026-09-20-fan-write-all-wta.md`). Queued
  ahead of this arm; it answers a different question (a selector with evidence from the
  span's own tokens) and does not test whether the one-stream write can transfer.
- **Do nothing and read the write-all arm alone.** Leaves the select family's deployed
  number confounded by the mismatch, which Wolfe named as possibly "part of our overall
  problem". Rejected.

## Acceptance criteria

The prereg's filing rule: P-1 (train-side and eval-side regret within 0.02), P-2 (paired
depth-6 gap against pk4 below +0.030) and P-5 (no written share above 0.70 or below
0.05, rank above 2.0). All three hold: the select family's deployed write transfers and
the note moves to `implemented/`. P-2 without P-5: width, not selection. P-1 and P-5
without P-2: the mismatch was not the deficit, the select family closes on the deployed
write, and this note moves to `rejected/`.

## Risks

- **The closed loop collapses the fan.** The gate picks, the coda learns those picks, the
  table then favours what the coda reads best, the gate's label follows. A fixed point at
  one stream is pk4 with a dead fan. P-5 reads it; the 1500-step anneal is the only guard.
- **Scheduled sampling exposure bias in the other direction.** For the first 1500 steps
  the coda still sees the table's winner in most slots; if the coda's reading is set
  early, the anneal buys nothing and the arm reads like select. P-1 separates the cases.
- **The gate's label carries 5 % noise** (unchanged from the filed arm). Named, not fixed.
