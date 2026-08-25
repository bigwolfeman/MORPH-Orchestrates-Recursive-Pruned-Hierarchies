# Agent Note: SCSE source-centered recurrence for the TUL core loop

Status: rejected — the forcing-bias mechanism SCSE removes is measurably absent from MORPH's
loop. A second argument against the port, that it would remove a de-correlator, was made here
and is WITHDRAWN: measured directly, the once-only source handling raises state diversity at 10
of 11 rungs.

## Problem

TUL arm A1 loops the weight-shared core over slot positions and diverges: validation CE bottoms
at step 500-2000 and then rises 1.3 to 2.4 nats. Arms A0 (loop over tokens) and A3 (no core) are
healthy at identical settings. The forward signature is a rank collapse of the slot states across
the loop.

SCSE (arXiv:2607.27656) describes a failure that fits that shape exactly. A recurrence that
re-injects its source at every step, `h_{t+1} = G_theta(h_t + e)`, leaves a source-driven term
`b_t(e) != 0` alive at the anchor, so "do nothing" is not a fixed point, the state drifts every
step, and the drift compounds with depth. MORPH's core step is that form twice over
(`morph/model/transformer.py:_apply_core_step`): a `DiagonalInjection` `A*h_ctx + dt*e_ctx` on
320 of 1024 channels, then `n_core` additions of a loop-invariant `inj_term_i` before each block.
`inj_term_i` does not depend on `h`, so `G_theta(0) != 0` by construction.

## Proposal

The proposal, had it survived: re-parameterise the core loop around a learned anchor. Use `e`
ONCE to produce `h*(e)`, evolve the deviation `Delta_t = h_t - h*(e)` through a bias-free core
constrained so `G_theta(0) = 0`, and mask the update so `Delta = 0` is an exact fixed point of
every iteration while non-zero deviations still move.

It is rejected. The structural match is real and is not evidence.

`lab/divergence/drift_probe.py` replays the captured core trajectory at all 11 onset-ladder
rungs (`checkpoints/morph/onset-capture`) and measures the per-iteration displacement
`d_t = f_theta(h_t) - h_t`. Full record and artifacts:
[../../../../docs/experiments/failures/2026-08-24-tul-zero-deviation-forcing-bias.md](../../../../docs/experiments/failures/2026-08-24-tul-zero-deviation-forcing-bias.md).

* The shared component of `d_t` does not accumulate across the loop. It DECAYS about 100x:
  `C_last / C_first` is 0.0076 at the TAKEOVER rung, where the pre-registered prediction needed
  at least 3.
* Zeroing the repeated additive injection — the exact term SCSE's `G_theta(0) = 0` constraint
  removes — moves the last iteration's shared concentration from 1.13 to 1.12.
* Removing the fresh per-step injection alone — `dt = 0`, the decay `A * h_ctx` left running —
  raises shared displacement concentration at 22 of 22 rung-path pairs. This was read as "the
  injection de-correlates, so SCSE would hurt", and that reading was WRONG. With the decay still
  running, `dt = 0` lets the ctx band decay toward zero, so the positions lose their identity by
  ERASURE. SCSE does the opposite: it holds each position's identity in a persistent anchor.

* The faithful counterfactual — source injected at iteration 0 only, no decay afterwards, run
  over the whole loop at every rung — goes the other way. Centred unit-direction effective rank
  of the slot states at the last iteration, on a fixed set of 96 positions: 18.79 to **30.83** at
  rung 1625, 11.27 to **24.71** at rung 1700, 25.02 to **29.86** at TAKEOVER. Higher at 10 of 11
  rungs, tied at the eleventh. SCSE's source handling would spread the slot states FURTHER apart,
  not merge them.

The third point is the decisive one for the port. SCSE replaces per-step injection with a
once-only anchor. In MORPH that would delete the only term measured to de-correlate the slot
states, in a model whose failure mode IS the slot states merging.

## Alternatives considered

* **Port SCSE as published** — anchor `h*(e)`, evolve `Delta_t`, zero-preserving core, zero-
  deviation mask. Rejected on the measurement above. It also arrives with the smaller caveats:
  139M parameters on WikiText, no stochastic depth, no truncated BPTT, none of which MORPH shares.
* **Take only the `G_theta(0) = 0` half** and keep per-step injection. This is what zeroing
  `inj_terms` approximates, and it changes the measured quantity by under 1 %.
* **Widen `DiagonalInjection` to all 1024 channels** — the opposite move, more injection rather
  than less. Consistent with the de-correlation finding and still open, but H15 already showed
  the existing anchor is healthy (`dt/(1-A)` is 1.8015 to 1.8047 across every rung), so more of a
  healthy thing is a weak prior. Kept in the ledger, not promoted.
* **Do nothing and keep looking.** Chosen, with the campaign's attention moved to the first core
  iteration — the one arm-specific forward asymmetry this run produced (A1's `C_first/P` is
  0.44-0.58 against A0's 0.17-0.19).

## What changed after this note was first written

The de-correlation argument was withdrawn the same day, after the ablation behind it was found
not to model SCSE at all. Two things follow, and they point in opposite directions, so both are
recorded:

* Against the port: its DIAGNOSIS is absent. That is the surviving reason, and it was always the
  stronger one.
* For the port, or at least against dismissing it: its source handling measurably raises state
  diversity in MORPH's loop. That benefit is probably worthless here, because the same run showed
  diversity is NOT the failing quantity — the loop raises diversity at every rung and raises it
  most at TAKEOVER, where the model is worst.

Do not re-derive the de-correlation argument. It was measured and it is false.

## Acceptance criteria

These were the gates the port would have had to clear, and the first one is what failed.

1. The mechanism is present: the shared component of the per-iteration displacement grows
   across the loop and grows across the onset ladder. **Failed** — it decays about 100x, on
   both arms.
2. The term SCSE removes is load-bearing: zeroing the repeated additive injection changes the
   shared component by at least 30 %. **Failed** — 1.13 to 1.12.
3. The mechanism is arm-specific: the healthy token path does not show it. **Not reached** —
   the control matches the failing arm on every trend, so no verdict about A1's failure could
   have been read from gates 1 and 2 even had they passed.
4. The port would not remove something MORPH needs. **Passed** — the once-only counterfactual
   raises state diversity at 10 of 11 rungs. This gate was added after the first version of this
   note asserted its failure without testing it.

## Risks

The first version of this note carried a second, untested argument and stated it with the same
confidence as the measured one. That is the risk this section exists to catch and it was not
caught. The general lesson is recorded in the campaign's trap list: an ablation that removes a
term is not a model of an alternative that REPLACES it.

The remaining risk is that SCSE is right about a MORPH failure this probe cannot see.
The probe measures displacement geometry over positions in one forward pass at a fixed depth
draw; it does not measure the loop's fixed-point structure over training, and it cannot see a
drift that is common to every position AND every example, because such a component sits inside
the mean it subtracts. That second gap is the one worth naming: `C` is computed against the
pooled mean over rows and positions, so a bias identical everywhere reads as concentration, but
a bias that is identical everywhere AND constant across the whole batch is still counted. The
de-correlation result does not depend on it — that one is an ablation, not a statistic.

Re-open this note if a later probe finds the loop's shared drift growing rather than decaying.

## Consequences

* SCSE is off the candidate list, with the number that removed it recorded. It was the top-ranked
  architectural candidate before this measurement.
* A new fact constrains every future candidate: the loop stops settling at the onset on the
  FAILING arm and the HEALTHY arm alike — `rel_last` rises 0.702 to 1.081 on slots and 0.682 to
  1.077 on tokens. Any cure whose mechanism is a contractivity restoration must explain why arm
  A0 never needed it.
* `lab/divergence/drift_probe.py` is reusable for any later candidate: it gates on reproducing the
  captured trajectory, so a probe that silently measures a different operator now fails loudly.
