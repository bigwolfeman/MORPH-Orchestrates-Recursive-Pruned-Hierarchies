# Agent Note: Source-centered state evolution for the MORPH core loop

Status: proposed

## Problem

MORPH's core loop is an additive-injection looped Transformer in exactly the sense
SCSE defines (arXiv:2607.27656). Two facts from the code and one from the paper's theory
combine into a statement much stronger than "MORPH has the shape the paper describes".

**Fact 1 — MORPH's initial deviation is exactly zero.** Both core paths start the loop with
`h = e.clone()` (`transformer.py:1043` for the token path, `:1359` for the TUL slot path).
With the natural input-conditioned anchor `h* = e` — a reference computed once from the input
and unchanged by the loop, which is the paper's definition — the initial deviation is

    Delta_0 = h_0 - h* = 0.

**Fact 2 — the forcing bias is large and grows with the onset.** Measured at all 11 rungs of
`checkpoints/morph/onset-capture` (`lab/divergence/drift_probe.py`, trajectory gate `0.0e+00`):
`||b||/||h*||` is 1.809 at rung 1625 rising to 2.471 at 1850 on the slot path, against 1.448
to 1.723 at the token anchor on the SAME weights. The shared transition moves the anchor by
about twice the anchor's own norm, at every recurrent step.

**Consequence — the entire loop trajectory is the propagated forcing response.** The paper's
decomposition (eq. 7-8) is `T_t(Delta) = Delta + b_t(e) + a_t(Delta; e)` with
`a_t(0; e) = 0` by definition. The bias-subtracted counterfactual (eq. 10) is
`Delta_bar_{t+1} = T_t(Delta_bar_t) - b_t` from `Delta_bar_0 = Delta_0`. With `Delta_0 = 0`:

    Delta_bar_1 = T_0(0) - b_0 = b_0 - b_0 = 0,

and by induction `Delta_bar_t = 0` for every `t`. So the counterfactual forcing response is

    E_T = Delta_T - Delta_bar_T = Delta_T,

the WHOLE deviation trajectory. There is no separate off-anchor computation to be spared:
every state the core loop visits is `h* + (propagated forcing response)`, which by Theorem 2
equals `sum_k Phi_E(T, k+1) b_k(e)`. Corollary 5 makes the growth explicit — if the secant map
has a real eigenvalue `rho` along `v`, then `<v, E_T> = <v, b(e)> * sum_l rho^l`. A `rho`
above 1 turns a bounded per-step bias into geometric growth over depth, and `rho(J_core)`
crossing 1 is the standing explanation for MORPH's looped-core instabilities
([iterative-map-dynamics](../../implemented/architecture/2026-06-19-iterative-map-dynamics.md)).

This is derivation plus measurement, not a training result. What it establishes is that MORPH
has no mechanism at all — no anchor separate from the initial state, no zero-preserving core,
no zero-deviation mask — bounding a quantity the paper shows can grow geometrically.

## Proposal

Adopt SCSE in four stages, each separately testable and reversible, ordered so that no stage
can freeze the loop. Reference implementation is Listing 1 of the paper; the update is

    Delta_t   = h_t - h*
    q_t       = s * G_theta(Delta_t)                        # s = 0.50
    m_{b,t}   = 1{ ||Delta_t^{(b)}||_F^2 > eps }            # eps = 1e-8, PER EXAMPLE
    Delta_t+1 = Delta_t + m_{b,t} * q_t
    h_T       = h* + Delta_T

with `h* = e + 0.1 * a_omega(e)` and `h_0 = e + 0.1 * H_0(e)`, both learned, both applied ONCE.

**Stage 0 — no model change.** Run an A0 ladder through the onset region with `b_t` and `R_t`
logged. H20 currently compares the slot anchor against the token anchor on arm-A1 weights; it
is not sick-against-healthy. About 40 minutes of GPU. Gate: does a healthy arm carry the same
`b_t` growth?

**Stage 1 — `H_0` only. `Delta_0 != 0`, additive loop untouched.** Replace `h = e.clone()` with
`h = e + 0.1 * H_0(e)` for a learned `H_0`, and keep everything else. This is the smallest
change that breaks Fact 1: the loop gets a state of its own that is not the propagated forcing
response. It is the paper's "tuned adapter" family, whose `R_47` falls from 4.351 to 1.619 with
PPL between the baseline and SCSE (their Table 3, Table 1). About 20 lines. Gate: `R_t` and CE.

**Stage 2 — deviation coordinates.** Loop on `Delta` instead of `h`; add `a_omega`; reconstruct
`h_T = h* + Delta_T`. The two per-iteration source injections move OUT of the loop and INTO the
anchor: `a_omega(e)` absorbs the `DiagonalInjection` steady state and the sum of the per-core
layer `inj_term_i`. See "what must not happen" below. Gate: CE against Stage 1 at matched steps.

**Stage 3 — exactness.** Audit the core for additive bias terms so `G_theta(0) = 0` holds
before masking, then add the per-example zero-deviation mask. RMSNorm maps zero to zero and the
attention sink adds a logit rather than an output offset, so this is expected to be an audit
rather than a rewrite — but it is an audit, not an assumption.

## Alternatives considered

* **Port SCSE wholesale in one change.** Rejected on risk. Stage 2 alone rewires the embedding
  injection paths, the loop carrier, and every divergence instrument that hooks
  `_apply_core_step(h, e, ...)`. Staging costs one extra arm and makes each result attributable.
* **Stage 1 only, and stop.** Genuinely tempting: it is 20 lines and it removes the structural
  claim above. It leaves `b_t` large, so the propagated response still dominates a trajectory
  that now merely starts somewhere else. Worth measuring before deciding.
* **Widen `DiagonalInjection` to all 1024 channels.** The opposite move — more source injection.
  Recorded in the campaign ledger; H15 already showed the existing anchor is healthy
  (`dt/(1-A)` flat at 1.8015-1.8047), so more of a healthy thing is a weak prior.
* **Do nothing; the diagnosis is refuted.** This was the earlier position
  ([rejected note](../../rejected/architecture/2026-08-24-scse-source-centered-recurrence.md))
  and it was built on the wrong quantity. H19 tested coherent accumulation of the SHARED
  component along the trajectory and refuted it; the paper's harm condition is coherent
  accumulation through the propagated secant maps (Corollary 4), which is a different test and
  is not run.

## Acceptance criteria

1. Stage 1 raises `Delta_0` off zero and `R_t` moves toward the paper's adapter band.
2. Stage 2 does not lose the bigram, `x0`, or value-embedding signal: the anchor carries them.
   Checked by an equality test on `a_omega(e)` against the current `inj_term` sum at init.
3. `G_theta(0) = 0` verified numerically before the mask is enabled, not assumed.
4. CE at matched steps, against the `g6-ctrl` control, with the control run first.
5. Every stage keeps `bptt_depth` at 4 and does not vary loop depth (both are standing vetoes).

## Risks

**What must not happen — the loop freezes.** MORPH is at `Delta_0 = 0` today. The zero-deviation
mask sets `m = 0` whenever `||Delta||_F^2 <= eps`, so enabling the mask BEFORE `H_0` exists
gives `Delta_1 = Delta_0 = 0` and the core makes no update for the rest of the unroll, forever.
The paper measured exactly this: its `h* = H_0(e)` anchor ablation collapses `Delta_0` to zero
and reports PPL 294.37 against the learned anchor's 155.14, "independent of loop depth"
(their Table 2). **Stage 3 must never run before Stage 1.** This is the single ordering
constraint that turns a working model into a dead one.

**Do not delete `inj_term_i`.** It carries the hash-bigram, `x0`, and value-embedding signal
into every core layer, and those paths are load-bearing in MORPH's hybrid-embedding design.
Fold them into `a_omega` so the information survives; removing them is a different experiment.

**`s = 0.50` is tuned for a one-block core.** MORPH applies six blocks per recurrent step, so
the per-step gain is not comparable and `s` may need its own sweep.

**Attention runs on deviations inside the loop.** Slots would attend to each other's deviations
rather than their states. The coda still reads `h* + Delta_T`, so the readout is unchanged, but
the in-loop semantics are not, and TUL routes the whole plan through those slots.

**Instrument breakage.** `core_jacobian.py`, `jac_ladder.py`, and `drift_probe.py` all hook
`_apply_core_step(h, e, ...)`. They need a deviation-coordinate mode or their numbers will
silently describe the wrong operator.

**Checkpoint incompatibility.** `a_omega` and `H_0` are new parameters. Fresh runs only.

**Evidence limits.** SCSE is evaluated at 22M-139M on WikiText-2/103, OpenWebText and C4, with
uniform depth sampling from {1..8} and no truncated BPTT, no stochastic depth, no ternary QAT,
and no structured sparsity. MORPH has all of those.
