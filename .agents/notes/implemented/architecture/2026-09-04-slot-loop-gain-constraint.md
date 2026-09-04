# Agent Note: the slot-loop gain constraint ships on by default

Status: implemented

Date: 2026-09-04. Branch `tul/think-once`. Wolfe: "let's ship the constraint to main and
document it clearly (another break glass in case of divergence doc with a long name so
it's easy to find)." Break-glass doc:
`lab/divergence/BREAK-GLASS-IN-CASE-OF-DIVERGENCE-THE-SLOT-LOOP-GAIN-CONSTRAINT.md`.

## Problem

Every slot-loop arm trained at full BPTT with ternary QAT on the winner recipe detonated
after the LR ramp (7 of 7 unregularised draws across three targets), and the onset capture
measured why: the map's typical gain drifts 0.87 → 1.00 and the run dies at the crossing.
A backward clip alone did not save it. The loop's stability was the blocker in front of
every contribution question.

## Decision

`morph/configs/base.yaml` carries `slot_gain_lambda: 100`, `slot_gain_target: 0.9`,
`slot_gain_eps: 0.02` and `slot_cot_clip: 4.0` for every model with a slot loop: the exact
configuration that took M-next to 5000 steps with zero spike steps and held the
fp32-measured gain at 0.887–0.897 for the whole run. `slot_state_renorm` stays off (the
other measured lever; either one, not both, was tested). A model without a slot loop
builds and prints `[slot-levers] ... INERT` once; all knobs at 0 / false are bit-identical
to the tree before them.

## Alternatives considered

- **Ship the renorm instead** (same survival, no extra passes, 116 against 110 steps/min).
  Not chosen: the penalty acts on the measured quantity and its trace (`gain_est`) is the
  instrument the next arms read; the renorm pins a proxy and its raw map still drifted to
  0.95. Both stay in the tree.
- **Ship the penalty without the clip.** Untested; the measured configuration wore the
  clip, so that is what ships. Removing the clip is a one-arm prereg when someone wants
  the cleaner default.
- **Raise at build when the levers cannot act** (no slot loop). Rejected once base.yaml
  carried them on: a coreless TUL arm and the plain model must build. A loud notice
  instead, so a run that thinks it is protected and is not can be caught in its log.
- **Port the constraint to the paid loop's core region for master.** Not done here: the
  paid map's healthy typical gain is 1.05–1.13 (the A2 ladder) and a target below its
  healthy value would fight a healthy run; that port needs its own prereg.

## Consequences

- Slot-loop arms on the recipe survive the plateau; the contribution question is now
  askable at 5000 steps and beyond (`2026-09-04-loop-contractivity-as-design.md`, open).
- ~5 % wall clock on slot-loop arms; two extra core steps of activation memory on the
  compact sequence per step.
- The penalty regularises one random iteration per step; iteration 0's gain still drifts
  (0.94 → 0.996 over 5000 steps). Known, documented in the break-glass file, unaddressed.
- Master (as of `3a94963`) has no slot loop: the paid loop was shipped on 2026-09-03 and
  `_tul_core` was cut. Landing this on master means either merging this branch (the slot
  loop returns) or porting the constraint to `_core_region`; that is Wolfe's call and is
  recorded here as open.

## Where it landed (2026-09-04, later the same day)

Merged into master with the whole branch — see the amendment in
[2026-09-03-ship-the-paid-loop-cut-the-arms.md](2026-09-03-ship-the-paid-loop-cut-the-arms.md).
On master the knobs sit in `base.yaml` under `model:` for every model; they act only in
`_tul_core`, which only `tul.tokens_through_core: false` (the `tul_short.yaml` family)
reaches. The shipped paid loop prints `[slot-levers] ... INERT` once at build and is
bit-identical to the pre-merge master (`lab/divergence/paid_loop_gate.py`).
