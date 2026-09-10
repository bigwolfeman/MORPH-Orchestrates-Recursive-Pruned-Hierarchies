# Agent Note: What a toy slot loop says about update flow in the real one

Status: proposed

Date: 2026-09-10. Study, code and every number:
[`lab/toy_slot_loop/WRITEUP.md`](../../../../lab/toy_slot_loop/WRITEUP.md); predictions frozen
before the runs in [`lab/toy_slot_loop/PLAN.md`](../../../../lab/toy_slot_loop/PLAN.md).
Sibling note: [`2026-09-10-credit-assignment-in-the-slot-loop.md`](2026-09-10-credit-assignment-in-the-slot-loop.md).

## Problem

Three measurements on the real slot loop box the failure in but cannot separate it from a
fourth explanation. The per-pass cotangent is flat, so gradient reaches every pass; the coda's
CE with the exit equals its CE with the entry, so the writer produces nothing the reader
wants; the six passes' updates to the shared core cancel at 0.520. On web text none of that
distinguishes "the attachment is wrong" from "the M-next target does not need iteration", so
every lever tried since 2026-09-04 has read flat without saying why.

A toy fixes the iteration requirement by construction. `lab/toy_slot_loop/` is a 0.60 M parameter model of the same shape — prelude, one shared core block looped on slot cells only,
prefix write, coda — on two span tasks: a non-commutative S_3 prefix scan whose answer at the
head of span j+1 provably needs j passes, and a one-pass control. 110 cells, 5 seeds each,
0 failures, on the 3070. The per-pass tap self-check (`sum_t dW_t == leaf.grad`) reads a worst
relative error of 1.63e-07 across all of them.

## Proposal

Three changes to what the tree does next, in order of cost.

**1. Read the cancellation ratio with the opposite sign.** Stuck toy runs read 0.946 and a
mean pairwise cosine of +0.845; runs that solved the chain read 0.749 and +0.443, and the
within-cell comparison holds in 7 of the 8 cells that produced both outcomes. The best arm in
the study reads 0.291, below the 0.408 that six orthogonal updates would give. Six passes of a
working chain have six different jobs; agreement measures how close a shared map is to doing
one thing six times. Acceptance criterion P-d in the credit-assignment note asks the ratio to
RISE above 0.60; on this evidence it should FALL.

**2. Check that `core_fixed_point_lambda` is binding in `_tul_core`.** The ruler
`slot-mux-norm-match` reports the term at 0.00302 against a loss of 11.28, and the A1 arm at
0.00023. In the toy the same-shaped term at lambda 1.0 took escape from 2 of 5 seeds to 5 of
5, turned a flat per-pass cotangent into a rising ramp (0.072 to 0.386) and drove the per-pass
cosine to -0.057. This is the cheapest lever in the report: raise it until it reads.

**3. Run the staged MUX target under `norm_match`, with `mux_stage_own_iters` set to the
loop's maximum depth** so every non-final pass carries the own-span target and only the exit
carries the next-span target. It is the only attachment of six that solved the scan, and it
did so on 5 of 5 seeds while exit-only did 2 of 5 and dense per-pass supervision did 1 of 5.
The knob exists; the two configs that use it (`tul_to_mnext_y2_stage2/3.yaml`) are from the
2026-09-04 lineage at k = 2 and 3 and predate the ternary rule change.

Bars at 5,000 steps: `mux_local` K1-K6 > 0.02 with K3-K6 > 0.002; at least one pass with a
negative cosine to the total in `slot_gradient_probe.py`; and the combined cancellation ratio
below the ruler's 0.520.

## Alternatives considered

- **Dense per-pass MUX on a live carry (`tul.mux_every_pass`, the arm running now).** Kept as
  an arm because it is already launched, but the toy calls it the worst live-carry attachment:
  1 of 5 escapes, a K-curve that stops one span past the shortcut, a per-pass cotangent that
  decays 7.6x toward the exit and a pairwise cosine of +0.966. Asking pass 1 for an answer no
  map can produce in one pass makes every pass ask for the same thing. **Prediction: that arm
  reads flat or worse than the ruler, with the cancellation ratio rising.** If it earns depth,
  this note is wrong.
- **Deep supervision through the coda (TRM/HRM).** Measured identical to exit-only to four
  decimals on the same seeds at 2.6x the wall clock. Rejected as an arm.
- **Bansal's progressive no-grad prefix.** 0 of 5 in the toy, the only attachment strictly
  worse than exit-only, and already filed flat on the real model. Rejected.
- **Per-pass low-rank deltas.** 0.7692 against exit-only's 0.7693 on the same seeds. The real
  sibling arm is running; the toy expects nothing from it.
- **Detached carry between passes (the DB style).** Reaches one span further than any other
  non-staged arm and then stops dead on all 5 seeds. It removes the only term that tells a
  pass to prepare a state for the next one, so it can learn one link and structurally cannot
  learn two. Not proposed, but worth keeping as the instrument that isolates that term.
- **Attacking the loop's geometry instead of its loss.** Deliberately NOT one of the three
  proposals above, because it is a different question and mixing it in would make the
  attachment result unreadable — but it is the largest single effect in the study and it is
  recorded here so it is not lost. Holding the attachment at exit-only and switching from the
  toy's strict geometry (loop is the only cross-span path) to MORPH's own (prelude causal over
  everything, core over every earlier slot, coda over every earlier prefix cell) moved the
  loop's contribution from +0.514 to +0.106 nats and made ONE core pass worth 0.542 on a task
  that formally needs six. Under MORPH's geometry the scan task behaves like the one-pass
  control. The cheap real-model test is the existing `tg_restrict` family, read as a K-curve
  rather than as a CE.

## Acceptance criteria

- The staged arm clears all three bars above at 5,000 steps, or it is filed flat and the
  attachment lane closes.
- `core_fixed_point_lambda` is shown to read a value that can shape the map, or it is raised
  until it does, or it is removed from the slot loop as inert. A term at 0.003 is not a
  hold.
- The `mux_every_pass` arm's filing states its cancellation ratio against the ruler's 0.520,
  so this note's prediction is scored rather than forgotten.
- The toy's own contracts stay green: `python lab/toy_slot_loop/selfcheck.py` passes all 47 checks, including the two that PROVE the depth requirement by differentiating the value
  logit with respect to an unreachable input cell.

## Risks

- **Transfer is unproven and the toy is 450x smaller.** 596,928 parameters, fp32 dense weights,
  12 vocabulary ids, one ordinary attention block in the core, 4,000 steps. MORPH is 268 M
  parameters with a ternary STE core, CCA + CSA + HCA + XSA attention and a 4-stream Cayley
  carrier whose slot states sit at effective rank 1.7-4.8 in 1024 dimensions. Nothing here is
  evidence about that.
- **The toy's outcome is bimodal, so its metric is an escape rate over 5 seeds.** A 2/5
  against 3/5 difference is noise and is labelled as such in the write-up. Only the extremes
  (5/5 for staged and for the fixed-point term, 0/5 for progressive and the noise entry) are
  read as results.
- **The synthetic target has no irreducible noise and an exactly known depth price.** Natural
  text has neither. The toy says what happens when a target needs iteration; it cannot say
  whether the M-next target on web text needs any.
- **The cancellation reading could be a scale artefact.** The toy's stuck runs sit at 0.95 to
  1.00 on 110,784 core parameters; the real model reads 0.520 on 68 M, where noise alone lowers
  the ratio. The DIRECTION is what transfers, if anything does, not the value.
- **Seven of fourteen frozen predictions failed**, including three of the ones that shaped
  this note (the fixed-point term, the sign of the cancellation ratio, truncated BPTT). The
  study earned its conclusions by contradicting its author, which is the good case, but it
  also means the priors going in were poor and the priors coming out deserve the same
  suspicion.
