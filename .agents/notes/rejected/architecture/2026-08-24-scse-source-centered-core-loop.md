# Agent Note: Source-centered state evolution for the MORPH core loop

Status: rejected — the FULL method was built, audited twice, machine-checked in Lean, and measured on 3 paired seeds: +1.68 nats mean CE against the control, 0 of 3 pairs improved, and the SCSE arm stops learning at step 200. The refuter fired.

## Outcome, 2026-08-25 — the full method was RUN and does not transfer

This supersedes the 2026-08-25 Stage 1 rejection, which was withdrawn as an overreach and is
kept below for the record. The port has now had a fair test.

**What was measured** ([experiment](../../../../lab/experiments/failures/2026-08-25-scse-full-method.md)):

| seed | control | SCSE | delta |
| --- | --- | --- | --- |
| 1 | 4.6863 | 6.4277 | +1.7414 |
| 2 | 4.5303 | 6.1722 | +1.6419 |
| 3 | 4.5073 | 6.1715 | +1.6642 |

Three seeds inside 0.10 nats of each other. The pre-registered validity gate PASSED, so this
is a result about SCSE and not about a broken arm. The failure mode is a STALL: the arm tracks
the control to step ~250 and then stops learning.

**Why this verdict is trustworthy where the last one was not.** The Stage 1 rejection tested a
configuration the paper never reports and then recommended abandoning the port. This one:
implements the two things the abstract actually credits (the learned anchor and the
anchor-coordinate deviation recurrence); survived two adversarial audit rounds that found a
structural bug and five fake tests; carries a machine-checked Lean proof
([lab/scse-lean](../../../../lab/scse-lean/README.md)) that the recurrence recovers the paper's
stability regime and reduces EXACTLY to the published algorithm when the carry is an identity;
and pre-registered a refuter that could kill it, which fired.

**What is NOT concluded.** Nothing about SCSE as published. The paper evaluates 22M-139M models
on WikiText-2/103, OpenWebText and C4, with uniform depth sampling and without truncated BPTT,
stochastic depth, ternary QAT or structured sparsity. MORPH has all of those, a 6-block core
where the paper has one, and a HyperConnection carry the paper does not model. This is a
statement about this port on this model, at this scale.

**Leading explanation, still a hypothesis.** `Delta_0` enters at ~0.1x the anchor norm while
MORPH's core blocks are RMSNorm-pre-normalised, so the first core application rescales the
deviation ~80x; `sigma_max` then runs to 6.04 against the control's 3.03, and in the bf16 the
model trains in, one core step carries 41-71 % error against 4.5-7.7 % for the control. Three
follow-up experiments are listed in the experiment file, in value order, none run.

**The code stays in the tree at `scse_enabled: false`**, which builds no parameters and draws
no RNG, so a control model is bitwise identical to one built before SCSE existed. Revisiting
costs one config field, not a rebuild.

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
response. **CORRECTED 2026-08-25: this is NOT the paper's "tuned adapter" family.** The tuned adapter
keeps the learned anchor and a learned scalar source gain in an additive update; Stage 1 has
neither, and the paper reports no initial-deviation-only control. About 20 lines. Gate: `R_t`
and CE. **MEASURED AND DEAD** — see the Status section.

**Stage 2 — deviation coordinates.** Loop on `Delta` instead of `h`; add `a_omega`; reconstruct
`h_T = h* + Delta_T`. The two per-iteration source injections move OUT of the loop and INTO the
anchor: `a_omega(e)` absorbs the `DiagonalInjection` steady state and the sum of the per-core
layer `inj_term_i`. See "what must not happen" below. Gate: CE against Stage 1 at matched steps.

**Stage 3 — exactness.** Audit the core for additive bias terms so `G_theta(0) = 0` holds
before masking, then add the per-example zero-deviation mask.

**Audit done 2026-08-24 (code read, not yet a numerical check).** It comes back clean, which
makes this stage much cheaper than budgeted. Every additive parameter in the core is either a
logit inside a softmax or a coefficient in a linear mixing; every value-carrying projection is
`bias=False`.

| term | where | why `G(0) = 0` survives |
|---|---|---|
| `gate_up`, `down` | `transformer.py:348-349` | `MortarLinear(..., bias=False)` — the whole SwiGLU path is bias-free |
| `W_aKV`, `W_aZ`, `W_bKV`, `W_bZ` | `attention.py:271-277` | `nn.Linear(..., bias=False)` |
| `B_a`, `B_b` | `attention.py:274,279` | added to `Z_a`, which is a **softmax logit**; the output is `(softmax(Z_a) * C_a).sum()` and `C_a` is bias-free, so at `x = 0` it is `w * 0 = 0` whatever `B_a` is |
| `sink_logits` | `attention.py:475` | one logit appended before softmax; the values are bias-free linear in `x`, so a zero input still gives a zero output |
| `temp`, `alpha` | `attention.py:459,465` | multiplicative |
| `proj.bias` | `hyper_connections.py:139` | the only `bias=True` in the core. It sets the three `n x n` **mixing coefficient** blocks, and mixing zero streams gives zero |
| RMSNorm `scale`, `log_scale`, `ret_gate` | `norms.py:59`, `mhc.py:76,206` | multiplicative; `RMSNorm(0) = 0` |

So the mask is expected to be a belt-and-braces boundary condition rather than the thing doing
the work, which matches the paper's own statement that the zero-preserving core is the primary
reparameterization. **Still required before Stage 3 ships: a numerical check that a zero
carrier through the real core returns zero.** The GPU was training when this audit was written,
so it has not been run.

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
3. `G_theta(0) = 0` verified NUMERICALLY before the mask is enabled. The code audit under
   Stage 3 is necessary and not sufficient: it establishes there is no additive output offset,
   but only a forward pass on a zero carrier proves it.
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
