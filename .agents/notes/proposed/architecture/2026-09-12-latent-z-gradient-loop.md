# Agent Note: an energy the slot loop cannot descend in one pass

Status: proposed

Record: [`lab/experiments/planned/2026-09-12-arc-latent-z-gradient.md`](../../../../lab/experiments/planned/2026-09-12-arc-latent-z-gradient.md).
Follows [`2026-09-10-gradient-conditioned-slot-passes.md`](2026-09-10-gradient-conditioned-slot-passes.md)
and [`2026-09-11-span-decoder-target.md`](2026-09-11-span-decoder-target.md).

## Problem

`slot-mnext-gradpass` (filed 2026-09-11) is the only credit-assignment arm on this tree
whose exit beat the ruler's forecast — 0.017 nats — and it did so by changing what a pass
can SEE rather than what it is asked for. The mechanism works. The measurement that
followed says the energy did not:

> the ladder falls 0.385 nats from t0 to t7, of which **0.366 is the FIRST pass**; t1 to t7
> is 0.019 and passes 3 to 7 sit within 0.03 of each other, slightly rising.

That shape is not a tuning accident. The energy was `_tul_mux_loss(target="own")` — the
slot's own span scored against an ORDER-FREE geometric bag through the tied head. Its
minimiser is the span's weighted unigram marginal, and the prelude has already run over
every token of that span before the slot's loop starts. The target was reachable in one
pass because the target only ever asked for one pass of work.

Two things follow, and they are the whole of this note:

1. **An energy with curvature in more directions than a marginal has** should give pass 2 a
   place to go. That is a testable claim about the energy, not about the loop.
2. **Nobody has checked whether the slot state is scoreable at all.** Every reading on this
   tree asks whether the CODA uses z. If a linear probe cannot tell a good z from a bad one,
   then no energy has anything to descend and the honest lever is the TARGET, not the
   conditioning. The literature's answer depends entirely on how the latent was trained:
   ROC-AUC near 1.0 on Huginn-style recurrent trajectories (Latent Thinking Optimization,
   arXiv 2509.26314) and chance on Coconut-style token-only-supervised latents
   (arXiv 2510.12167, Cohen's d 0.17). MORPH's slot is the second kind.

## Proposal

**A gate, two energies, one control, one bound, one refusal.**

**The gate.** `lab/divergence/slot_state_linear_probe.py`: a cross-validated linear probe
on the per-pass slot state against "is the coda's mean CE over this slot's next span below
the median", with a shuffled-label null band, an entry-vs-exit control and the
`slot_z_optimize` split-point gate. It runs before the arms and it decides whether they are
worth reading.

**The energies** (`morph/model/tul_egrad.py`), both feeding the existing `TULGradPass`
conditioning path unchanged:

* `recon` — a second `SpanDecoder` with its own parameters, reconstructing the slot's OWN
  span from z, teacher-forced, with the A\*-Thought-V2 Label-Forcing soft target
  (`(1−m)·onehot + m·bag`). Conditional, ordered, per token.
* `disc` — `E = −s_phi(z, ctx)`, a 2-layer critic trained with BCE against the gate's own
  label, with shuffled-context negatives.

**The control** `slot-spandec-norecur` (`tul.slot_depth_fixed: 1`): the same parameters and
the same target with the recurrence removed, so "the loop iterates usefully" is separated
from "the slot cell and its extra map are useful". No new code.

**The bound** `tul.pass_residual_lambda` (0.01 on both arms): `λ · mean_t ‖h_{t+1}−h_t‖² /
‖h_t‖²` over the loop's gradient passes. Distinct from `core_fixed_point_lambda`, which
charges the same ratio at each slot's LAST pass only and stays at the ruler's 1.0.

**The refusal** `tul.reinject_seed_every_pass`: raises at construction. See below.

Three invariants hold by construction and by test, not by inspection:

* the energy's own training loss runs on a stop-gradient copy of z, so it reaches its own
  parameters and stops. `W_g` is the ONLY route from an energy into the loop.
* the feature crossing `W_g` is detached (`create_graph=False` and an explicit `.detach()`),
  so no second-order term exists.
* `grad_pass_energy: "own_mux"` — the default and every existing arm — builds neither
  module, draws no RNG and leaves the forward bit-identical.

## Amendment 2026-09-12: the arms moved onto the strict geometry

All three arms now compose `tul_slot_spandec_strict` and are named
`slot-spandec-strict-egrad-recon`, `slot-spandec-strict-egrad-disc` and
`slot-spandec-strict-norecur`. Nothing about the energies, the bounded residual or the
depth-1 control changed; the FORWARD they run on did.

The reason is a number already in this arc's record. On `slot-spandec-mask` at 5,000 steps,
480 rows, the whole slot channel (`worth_profile --plan-mode all_slots`) is worth **0.182**
nats and the loop's own prefix write (`zero`) **0.078**. More than half of what the slot
channel carries never passes through an iteration: under `tul.tg_restrict` the prelude lets
every token and every cell read every earlier cell, so a cell's SEED — a bag-mean of its own
span — reaches later spans with no pass of the loop in between. An energy whose entire claim
is about what the loop's PASSES do was therefore being scored on a forward where the passes
are optional.

`tul.tg_geometry: "strict"` closes those routes
([`2026-09-12-strict-slot-geometry.md`](2026-09-12-strict-slot-geometry.md)). The
one-factor partner of each arm is now `slot-spandec-strict`. The cost of the move, stated
rather than hidden: two frozen predictions (P-e, P-h in
[the prereg](../../../lab/experiments/planned/2026-09-12-arc-latent-z-gradient.md)) were
written against a `slot-spandec-mask` baseline and will be scored against a different
control; they are NOT edited, and the prereg's amendment 1 says so. The Step-0 probe gate
was also run on mask-geometry checkpoints and has not been run under strict.

## Alternatives considered

**Use the energy gradient AS the update** (`h_{t+1} = h_t − η ∇E`), the obvious reading of
"make the loop an optimiser". Rejected on the measured comparison the design digest carries:
a scalar-energy gradient used as the update is **+3.4** where a learned vector transition
CONDITIONED on that gradient is **+10.7**. `TULGradPass` is already the second form, which
is why this batch changes the energy and not the update rule. Building the first form would
also have discarded the one thing gradpass proved works.

**A JEPA-style regression target on z** (predict the next slot's state, or a pooled span
embedding, from z). Refused outright by the spec's standing rule — never regress onto the
slot state (LCM, CoCoMix, BT §4.2) — and the rule is not decorative here: a regression
target lets the loop win by collapsing z, which is the failure mode
`sigreg_lambda` exists to fight. The `disc` energy is the legal shape of the same idea: the
latent is an INPUT to a scorer, never a target.

**The second-order form** (`create_graph=True`, one Hessian-vector product per pass, the
learned-optimiser objective of Andrychowicz et al. 2016). Deferred, not rejected. It is the
binding clause's own next twin from the gradpass filing, and running it at the same time as
a new energy would change two things at once. It is the first follow-up if `egrad_t` keeps
falling past t1 and the coda still reads nothing.

**Sharing `tul.spandec`'s decoder as the energy** instead of building a second instance.
Rejected: the target decoder grades the NEXT span and IS the loss the loop is judged on, so
conditioning the passes on its gradient would hand the loop the answer — that is exactly
the `slot_z_optimize` ORACLE, not a mechanism a generator could run. The two decoders also
now take different private init offsets so they do not start from identical weights.

**Fitting the energy at every pass instead of at the exit.** More principled — the energy
would then be a potential over the whole trajectory rather than one fitted at its end — and
roughly double the arm's extra FLOPs. Taken as a cost trade, named in
`_egrad_train_loss`'s docstring and in the prereg's "Not verified".

**`egrad_max_tokens: 32`** (the full span cap, matching `spandec_max_tokens`). Rejected on
arithmetic: the energy's head readout is paid at EVERY pass while the target's is paid once,
so J = 32 is ~90 % of a step against J = 8's ~22 %. 8 is also where the measured cross-span
budget stops being front-loaded.

**Sampling the soft label instead of computing it.** A stochastic hard label whose
expectation is the mixture would have been one CE call instead of two. Rejected: it puts
noise into a feature that conditions the loop, and the exact form costs only a second call
to a kernel the tree already has, because CE is linear in the target.

**Building `reinject_seed_every_pass`.** Rejected because it is already true. `_tul_core`
binds `_e_arg = e` — the prelude's output at the slot position, i.e. the slot seed
`E_slot + W_sent·embed(t_last)` after the prelude has run over it — ONCE, and hands it to
every pass; `_apply_core_step` opens with `self.injection(h_in, e_in)`, a
`DiagonalInjection` of that same `e` at every pass, then adds the per-core-layer x0/bigram
terms gathered at the slot positions. LRT's −3.2 points for an init-only seed is a real
finding about a design MORPH does not have. A second additive copy would be a duplicate
path no measurement could separate from a change in the injection's gain, so the knob exists
and RAISES with that explanation — discoverable, and impossible to switch on by accident.

## Acceptance criteria

1. **The gate is honest.** The shuffled-label control reads 0.50 ± 0.03 and the split point
   reproduces the trained forward's token CE bit-exactly, on every checkpoint the probe is
   run on. (Both hold on the two 6-row CPU smokes; the pooling defect that made the first
   smoke read 0.576 on shuffled labels is fixed and recorded.)
2. **Off is nothing.** `grad_pass_energy: own_mux` and `pass_residual_lambda: 0` give a
   bit-identical loss and bit-identical gradients on every parameter, and the plain path
   (`slot_layout=None`) is bit-identical with every new module built.
3. **The energy cannot shape z.** `autograd.grad(energy_training_loss, h_slots)` is None on
   both energies, and the injected feature carries no `grad_fn`.
4. **The arms are readable.** `loop/egrad_t{t}` and `loop/gp_rel_t{t}` are logged every
   step, so "the feature never acted" and "the energy was never descended" are separable
   from "the coda did not read the result" — the three-way split gradpass's filing needed.
5. **The arms survive to 5,000** with no sustained tripwire, and `recon` clears the 8,086
   tok/s rate floor at step 200.

## Risks

* **The gate reads chance and the arms still run.** Then both arms are uninformative rather
  than negative, and the honest next move is the target. Mitigated by running the gate
  first and by `egrad_auc`, which says the same thing from inside the `disc` arm.
* **The hinge does not bound the feature.** Its finite difference treats the injected term
  as exogenous, so a `W_g` that grows large adds gain the constraint cannot see. Inherited
  from gradpass, stated again because this batch adds a second unhinged input. `loop/
  core_gain_t0` and the sustained tripwire are the only watch.
* **Cost.** The whole `recon` estimate is arithmetic. If the inner backward through a
  2-block decoder launches badly on the eager TG-scoped path, the arm fails the rate rule
  and the queue skips it — which is the designed outcome, not a surprise.
* **`egrad_weight` is an untuned 1.0.** The energy's training loss sits in the same total as
  the token CE and the span decoder's. It is subtracted from every reported loss
  (`egrad_weighted`, the `spandec_weighted` precedent) so the arm stays comparable, but its
  size relative to the other terms is a guess.
* **Two new terms at once on each energy arm** (the energy's training loss and the residual
  bound). They are one factor *from the parent* as a pair, not individually. If an arm
  moves, a follow-up at `pass_residual_lambda: 0` is needed before the move is attributed
  to the energy.
