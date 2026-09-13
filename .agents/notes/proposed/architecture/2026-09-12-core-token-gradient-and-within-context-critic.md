# Agent Note: the core-token gradient auxiliary and the within-context critic

Status: proposed

Arms `slot-spandec-strict-coretok` (`tul.core_token_aux`) and `slot-spandec-strict-critic`
(`tul.grad_pass_energy: critic`), 2026-09-12. Pre-registration:
[`lab/experiments/planned/2026-09-12-arc-core-token-and-critic.md`](../../../../lab/experiments/planned/2026-09-12-arc-core-token-and-critic.md).
Both run on the strict geometry
([`2026-09-12-strict-slot-geometry.md`](2026-09-12-strict-slot-geometry.md)) and the
critic composes the gradient-pass machinery
([`2026-09-12-latent-z-gradient-loop.md`](2026-09-12-latent-z-gradient-loop.md)).

## Problem

Twelve slot-loop arms since 2026-09-04 read a token K1−K6 inside [−0.0001, +0.0033]. The
strict ruler reads 0.0016. The only arm that ever moved the curve without blinding the
coda is none; `prev-reach1` moved it to +0.0163 by amputating the coda's reach, and Wolfe
rejected that as an answer: *"z has to hold the history when it should hold the present
next thought that needs decoding. Our objectives are still poor."*

Two measured facts have not been acted on.

**Nothing trains the core.** In the slot loop the six shared core blocks see the SLOT
losses and nothing else: ~51 valid cells a row, one exit target each. The token CE reaches
the core at about 1 % of the prelude's gradient (the 2026-09-10 per-pass cotangent probe).
The PLAIN looped model trains the same six blocks on 1,024 next-token targets a row and
earns 0.185 nats of depth. The 2026-09-11 overnight batch closed the core as a SHAPE
question — nine levers, including a whole replacement architecture (Parcae), all flat —
but every one of those was measured on a core trained by the slot losses alone. **What
trains the core is the one input to the loop that has never been varied.**

**Every energy handed to the passes has been reachable in one step, and none was measured
against the coda.** `own_mux` fell 0.366 nats in the first pass and sat flat for seven.
`disc`'s label is "is this slot's next span below the BATCH MEDIAN CE?", which is mostly a
property of the text — which is exactly why it needs shuffled-context negatives to stop
the head scoring from the context alone.

Wolfe, 2026-09-12: *"train the core on the token CE as well (tokens through the core for
gradient only, slots still the only cross-span channel at inference), and a within-context
critic that scores whether the state after pass t beats the state after pass t−1 on the
same coda loss."*

## Proposal

### `tul.core_token_aux` — the token CE through the core, in TRAINING only

After the shipped forward finishes, a SECOND pass over the same prelude output sends every
position — tokens and slot cells together, the paid loop's shape — through `_core_region`
(the per-sample Poisson-depth core the plain model and the paid loop already run; **no
second core is written**) and then through `_back_region`, and charges `_tul_group_losses`'
ordinary weighted CE. The term is exposed as `core_token_aux` / `core_token_aux_weighted`,
which `train.py` subtracts, so `train/loss` and the val loss stay the shipped path's CE.

The SHIPPED forward does not change. The aux is guarded on `self.training`, so eval, the
forced-depth sweep, `worth_profile`, `slot_z_optimize` and inference all run the slot loop
with tokens OUTSIDE the core, and the slot cells remain the only channel that crosses a
span boundary at inference. **This is not the paid loop under another name**, and the test
that says so is `test_the_shipped_eval_forward_is_identical_on_and_off`.

Three decisions inside it, each with its reason:

**The aux core is restricted.** `causal AND (same span OR j is a slot cell)` on the window
branch, the same slot-column restriction on the compressed branch, the `tg_segment_ids`
reset on the CCA conv and its `W_v_prev` value shift. Without it the core would learn a
cross-span route the shipped forward does not have, and the arm would buy its gradient by
re-opening the bypass `strict` exists to cut. A token reads its own span and reaches every
earlier span only through a slot cell — the reachability the cells have inside the loop.
Proved by a leak test, not asserted.

**The depth is its own draw.** The slot loop draws a per-SLOT depth `[B, S]`;
`_core_region` draws a per-SAMPLE one `[B]`. Every reduction of the first to the second
distorts the distribution — the max over ~51 Poisson(6) draws capped at 8 is 8 almost
surely — so the aux takes the plain model's own per-sample Poisson draw. Training the core
the way the plain model trains it is the arm's entire claim.

**The aux is RNG-neutral and state-neutral.** The whole call sits in a save/restore of the
generator state (the `_slot_gain_penalty` precedent), so `loss − core_token_aux_weighted`
is the off-model's loss bit for bit whatever runs after it. `self._core_aux` is saved and
restored too — `_core_region` stashes its fixed-point term there and the slot loop has
already written its own; charging both would apply `core_fixed_point_lambda` twice a step
and make `fixed_point` incomparable with every earlier arm. `self._jac_capture` is disabled
for the duration, or a probe would record two different maps under one name.

`_core_region` gained the ability to thread `tg_seg` and `tg_comp_allow` through its
active-set sort (it already threaded `tg_allow` and `tg_slot_mask` for the A2s arm), and
now RAISES on any other attention kwarg instead of dropping it silently.

### `tul.grad_pass_energy: critic` — the within-context improvement critic

The conditioning path is byte-for-byte the `disc` arm's. What is new is the LABEL: for each
valid slot, two candidate states are each written into that slot's prefix cells through the
SAME `prefix_project`, the REAL coda is replayed with the SAME dropout `keep`, allow
relation and retention reset, and the next span's mean token CE is measured per slot.
Everything the two candidates share — the row, the context, the span, the decoder —
cancels, which is what "within context" buys and what `disc`'s batch-median label does not
have.

Three candidates, two pairs: the trajectory pair `(h_{t−1}, h_t)` at a per-slot `t` drawn
uniformly in `[1, realised depth]`, and the perturbation pair `(h_t, h_t + eps·rms(h_t)·n)`.
The perturbation is not decoration: without it the critic only ever sees states the loop
already produces and has no reason to be smooth anywhere else, which is exactly where its
gradient is read.

The loss is pairwise logistic on the score difference, weighted by the measured CE gap, on
DETACHED `z`. **Every replay is `no_grad`** — there is no graph at all, so "the label
trains nothing" is structural, not argued.

Two readings come with it: `critic_agree` (the honesty instrument, the `egrad_auc` twin — a
critic at 0.5 means the energy carries nothing) and `critic_gap_traj`, the mean
`CE(h_{t−1}) − CE(h_t)` through the real coda. That second one is the **measured worth of
one pass**, which every K-curve has read near zero and which no instrument on this tree has
measured directly.

`tul.critic_replay_groups: G` exists because the confound is real: under
`tg_coda_prefix_reach: all` a token of span `s+1` reads every earlier slot's cells, so
substituting every slot at once attributes span `s+1`'s CE change to slot `s` while every
earlier slot's substitution also moved it. G splits the substitution over G replays so the
nearest confounder sits G spans back. **G = 1 is the arm's setting and it accepts the
noise**, which is a cost decision and is written down as one.

### `tul.grad_pass_energy: "coda_exact"` is REFUSED, as a named value

The follow-up asked for — the energy IS the next-span coda CE, replayed at every pass — is
kept as a value that RAISES with its reasons, the `reinject_seed_every_pass` precedent, so
the refusal is discoverable from the config. It does not fall out of the critic's replay:
the coda's inputs do not exist yet when the energy is read inside `_tul_core`; the critic's
replay is `no_grad` by contract while `coda_exact` needs it differentiable; it would move
the token-state dropout's RNG draw, changing the shipped forward on every arm; and T coda
forward+backward passes per step is 36,864 block-passes against the model's own 11,052.

## Alternatives considered

**Arm C: share one depth draw between the slot loop and the aux core.** Recommended in the
brief and rejected on the arithmetic. The two draws have different shapes, and the only
reductions available (max, mean over a row's valid slots) distort the distribution — the max
over ~51 Poisson(6) draws capped at 8 is 8 almost surely, so the aux would train the core at
a depth the plain model never runs, which defeats the point. The aux takes the plain model's
own draw and the stream is restored afterwards, so nothing about the run's RNG moves.

**Arm C: detach the aux's entry so it trains the core alone.** This would make the arm one
factor in the strictest sense. Rejected: the aux coda would then be read on an input
distribution it never learned, so the aux CE would mostly measure that mismatch rather than
the core. The live version is kept and the confound is named in the pre-registration, with
`core_token_aux_ce` vs `val/ce_main` as its instrument.

**Arm C: run the aux over the TOKEN positions only, gathered.** Cheaper (1,024 instead of
1,152 positions) and wrong: the paid loop's shape is tokens and slots in ONE sequence, and a
gathered token axis changes what a cell's key is and which positions the CCA conv sees. It
would also need the allow relation and the segment ids re-derived on the gathered index
space — the same thing `_forward_tul` already REFUSES for `tg_restrict` with a gathered
coda.

**Arm C: run the aux BEFORE the shipped core instead of after.** Rejected: `_core_region`
overwrites `self._core_aux` and draws from the RNG stream, so running it first would move
the slot loop's own Poisson draw away from the ruler's at step 0. Running it last plus the
save/restore makes the arm differ from its ruler by the mechanism alone.

**Arm D: a direct regression of the CE difference instead of a pairwise logistic loss.**
Rejected: the CE gap's SCALE is not what the energy needs — only its sign and the confidence
in that sign are — and a regression would spend the head's capacity fitting how hard each
row is, which is `disc`'s measured failure mode. The CE-weighted pairwise form keeps the
sign and uses the magnitude only as a weight, so a pair the coda cannot tell apart teaches
nothing instead of teaching a coin flip.

**Arm D: use the exit state for the perturbation pair so one replay comes free.** It would
cut the cost from three replays to two, because the shipped coda has already scored the exit
state. Rejected: then the trajectory pair and the perturbation pair sit at different points
of the trajectory, and the critic would be fitted at the exit while its gradient is read at
every pass. Both pairs share `h_t` on purpose.

**Arm D: reuse `egrad` / `egrad_weighted` for the critic's term.** Cheaper in plumbing —
`_egrad_train_loss` already owns the energy modules' own losses and `train.py` already
subtracts that key. Rejected: the critic's label needs the coda's inputs and a replay of
`_back_region`, which that signature does not have, and sharing the key would put two
scorers with different labels on one wandb series. `_egrad_train_loss` returns `None` for a
CriticEnergy and says why.

**Arm D: make the critic the loop's UPDATE rather than its conditioning.** Rejected before
this change and recorded in `TULGradPass`: a scalar-energy gradient used AS the update is
measured weaker than a learned vector transition conditioned on it.

**Build `coda_exact` anyway.** See above — four missing pieces, one of them a change to the
shipped forward's RNG draw and one a 3.3× cost. The refusal names all four.

## Acceptance criteria

* `tul.core_token_aux: false` and `grad_pass_energy != "critic"` are BIT-IDENTICAL to the
  tree before this change: no module, no key, no RNG draw, and the methods never called
  (proved by monkeypatching them to raise).
* `loss − core_token_aux_weighted` and `loss − critic_weighted` equal an off-model's loss
  bit for bit, from the same seed.
* The aux term's gradient reaches EVERY core block, and reaches neither the span decoder nor
  `tul.W_prefix`.
* Inside the aux core, one application with the slot cells blanked moves nothing outside the
  edited span, bit-exact on CPU fp32 — with two two-sided controls that DO see a leak.
* An eval forward is identical with `core_token_aux` on and off, so every sweep and probe
  reads the same model.
* Every loop, coda, `W_prefix` and span-decoder gradient is bit-identical with and without
  the critic's label term, while the critic's own gradient reaches the passes.
* Both configs compose through Hydra at the panel's real budgets (seq 1024, batch 6), build,
  and run one training forward and backward.
* `python scripts/verify_template.py` adds no line.

## Risks

* **Arm C's memory.** The aux coda's activations are live at the same time as the shipped
  path's graph. `-perpass` died of exactly this class of surprise on 2026-09-12 and the
  priced cause was off by 10×. Named in the pre-registration as the most likely way arm C
  dies, and it will show up as an OOM in the smoke rather than as a slow step.
* **Arm C's rate.** 59.8 block-passes per real token against the strict ruler's 14.8 — by
  proportion, near 2,700 tok/s against the 8,086 paid-loop floor. The runner it is queued on
  has the floor OFF.
* **Arm C could become two models.** If the aux CE falls far below the model's own token CE,
  the core has learned a map the shipped forward never runs and no K-curve reading means
  anything. `tul/core_token_aux_ce` against `val/ce_main` is the guard, and it is the FIRST
  number to read on that arm.
* **Arm C is not one factor on the CE axis.** Its backward reaches the prelude and the coda
  as well as the core. The K-curve reading is still one factor (the shipped forward is
  untouched); the CE reading is not.
* **Arm D's attribution confound** at `critic_replay_groups: 1`, described above and
  accepted.
* **Arm D can be a null that looks like a bug.** If one pass genuinely changes the coda's CE
  by almost nothing, the labels are near-ties, the CE weighting shrinks the term toward zero
  and `critic_agree` sits at 0.5. That is a RESULT — `critic_gap_traj` is the number that
  says which it is — and the pre-registration binds on it.
* **`critic_every` counts TRAINING FORWARDS, not optimiser steps.** Under gradient
  accumulation those differ; the panel runs accumulation 1.
