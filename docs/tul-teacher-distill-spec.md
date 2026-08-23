# Arm D — teach the slot what to carry, by distilling a teacher that still has the tokens

Status: SPEC. Not implemented. Written 2026-08-18.

## The result this exists to fix

Arm CW, re-scored by stratum (docs/experiments/results/2026-08-18-tul-arms-first-comparison.md, 200 batches, 1.42M tokens,
cut 576). Splitting predictions by whether the target's last prior sighting is inside
the deleted region:

| stratum | share | slots (CW1) | equal-budget random tokens (CW2) | CW2 − CW1 |
|---|---|---|---|---|
| **A — only prior sighting is in the deleted text** | 15.8% | 3.4454 | **3.3466** | **−0.0988** |
| B — also present in kept context | 47.6% | 2.1347 | 2.1736 | +0.0388 |
| C — no prior sighting at all | 36.5% | 4.8794 | 4.8954 | +0.0160 |

**On the only stratum where memory can matter, keeping 128 RANDOM real tokens beats
keeping the slots by 0.0988 nats.** The slots' aggregate advantage lives in B and C —
strata where the deleted content is not needed, and where a summary cannot be acting as
memory. Stratum C was the pre-registered control and it fired.

Mechanism, once stated, is obvious: CW2 retains 128 of ~500 deleted positions, so about a
quarter of the time it holds the *exact* token being predicted. A mean over a span never
can. For the long-range dependencies this data actually contains — largely repetition —
an exact copy of a random quarter beats a gist of everything.

Nothing trained this slot to carry span content. Token CE is satisfied by the cheap
channel while the tokens are present, so no gradient ever asked the slot for identity.
This arm supplies the missing demand.

## The design

**Teacher = the same model, same weights, with the tokens still present. Student = the
same model with them deleted.** Both halves already exist and are tested:
`tul_forward_cw_arms` runs CW0 (everything kept) and CW1 (slots only) over one shared
front/core prefix, scoring an identical position set. CW0 IS the teacher, CW1 IS the
student. The distillation term is a new loss on an existing forward, not a new harness.

Self-distillation across the compaction boundary is exactly the objective we want stated
plainly: *your compressed state must induce the same downstream computation as having
everything*. No second model, no cross-model representation mismatch.

### Do NOT target the final layer

The last layer is collapsed onto next-token prediction — it is the most surface-shaped
state in the network. Distilling into it teaches the latent to be a next-token state,
which reproduces the exact bias this whole line has been fighting. Target **mid-stack**
residual states. Our own Future Lens work (`scripts/ltd_future_lens.py`, "probe mid-stack
onto a final-layer target, per Pal et al.") and the CODI note that "a last-layer-only
alignment would supervise one slice of a residual stream the injection reads whole" both
point the same way.

Which depth is not a guess — see Phase 0.

### Loss

At the shared scored positions (token positions with row index >= cut):

    L = CE_student + lambda * D(h_student_layer_l, stopgrad(h_teacher_layer_l))

* `D` = cosine distance + L2, **regression, not contrastive, as the primary form.**
  A contrastive loss (SigLIP-style sigmoid pairwise) optimises *discriminability*, not
  *sufficiency*: a state can be perfectly identifiable and still not support the
  next-token computation. That is precisely the trap the slot-collapse probe fell into on
  2026-08-18 — nearest-neighbour retrieval was well above chance (0.315 vs 0.038) while
  the same slot's contribution to loss was +0.027 nats. Keep the sigmoid-pairwise variant
  as a fallback if the regression target proves unstable, not as the default.
* `stopgrad` on the teacher. The teacher is the same weights but must not be trained
  toward the student — that is the collapse channel.
* **CE keeps its full weight.** Sweep `lambda`, do not reduce CE. Under masking the
  student's CE is already on a harder task; trading it away risks buying a good
  representation-matcher that is a worse language model. If the two objectives genuinely
  fight, that is a measurement, not an assumption.

### Cut schedule

Random cut per batch, ramped from late to early over training. A fixed cut teaches
"position 576 is special"; a random cut forces every slot to be ready to be the only
survivor. The ramp exists because a hard cut from step 0 makes the arm learn slowly for
reasons unrelated to slots — the "identical config is not an identical task" failure this
project has already paid for once.

## Phase 0 — the layer probe, before any training

Which depth carries the most information about the deleted region? Runs on
`tul-a1-acap1/step_20000.pt`, no training.

For each coda layer `l`, measure how well `h_l` at scored positions predicts content from
the deleted region — the cleanest form is the stratum-A CE of a linear readout, or the
mutual-information proxy already used by the Future Lens code. Report the curve over `l`.
Aim the distillation target at the peak.

If the curve is flat, say so: it would mean depth choice does not matter and the
final-layer objection was theoretical.

## Pre-registered success criterion

The bar is set by what CW already measured, and it is deliberately harsher than "beats
nothing":

> **Arm D succeeds if, on stratum A, the trained student with slots beats the
> equal-budget random-token control.** Today that gap is −0.0988 in the wrong direction.
> Success is crossing zero, measured the same way, same cut, same strata definition.

Beating CW3 (keep nothing) is NOT success — CW1 already does that (+0.0267 on stratum A)
and it means only that 128 positions of anything are better than none.

**The truncation control must itself be TRAINED.** Comparing a student trained for the
masked task against a baseline that only met masking at eval biases the result toward us.
That is a second training run, and it is the honest price.

## Falsifiers, stated before the run

1. **Distill loss falls, stratum A does not move.** Then matching the teacher's
   representation is achievable without carrying the information that matters, and the
   target is too easy. Report the distillation loss and stratum-A CE on the same axis,
   every run — a paper that reports only the first has proved nothing.
2. **Gains concentrate in strata B and C.** Same failure as CW: an aggregate win that is
   not memory. Always report stratified.
3. **The slot still cannot hold identity.** If stratum A stays negative even with the
   objective demanding it, then a d=1024 mean over a span cannot preserve which rare
   token appeared, and the answer is a copy/pointer-shaped mechanism rather than a larger
   or better-trained summary vector. That is a real finding and it retires this design
   rather than motivating a bigger one.

## Explicitly out of scope: reasoning

An earlier framing wanted the latent to carry a *thinking trace*, with a trace-inversion
model (arXiv:2603.07267) manufacturing targets. That is a different problem and it does
not belong in this arm.

OpenWebText contains no reasoning. A teacher run over it never reasoned, so there is no
trace to distill at any layer. Carrying reasoning needs a teacher that actually reasoned,
which means reasoning traces as data and a separate post-training phase — not a
pretraining objective bolted onto web text. Conflating the two is the "two goals, one
mechanism" error this project has already made once.

If that phase happens, the first check is not accuracy. It is: **does the inverter's
output vary with its input on OUR data distribution?** Feed it several samples and
compare. An inverter that emits plausible reasoning boilerplate regardless of input gives
a target carrying no information about the sample — it would look like a working
objective while teaching nothing.
