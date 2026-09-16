# Agent Note: TUL-Code — the slot holds the code of the span it precedes, and the loop samples it

Status: proposed

Date: 2026-09-14. Spec: [`docs/tul-code-spec.md`](../../../../docs/tul-code-spec.md) (the
contract; this note is the decision and what it gives up). Prereg to write before the first
run: `lab/experiments/planned/2026-09-14-arc-tul-code.md`.

Follows [`2026-09-12-strict-slot-geometry.md`](2026-09-12-strict-slot-geometry.md) (the
geometry it runs on), [`2026-09-11-span-decoder-target.md`](2026-09-11-span-decoder-target.md)
(the last target the loop was given), [`2026-09-12-latent-z-gradient-loop.md`](2026-09-12-latent-z-gradient-loop.md)
(the probe that says the state is scoreable at entry) and [`2026-09-14-loopmtp-token-loop.md`](2026-09-14-loopmtp-token-loop.md)
(the chain-versus-fixed-point reading). Stands against the rejected
[`2026-08-28-tul-fm-arc.md`](../../rejected/architecture/2026-08-28-tul-fm-arc.md) and the
binding rule in [`2026-08-30-objective-lines-vs-l2cap.md`](../../implemented/architecture/2026-08-30-objective-lines-vs-l2cap.md);
see "The binding rule" below. Supersession check: no active note makes a claim about what
the slot state IS that this one contradicts; the span-decoder note defines a TARGET for a
loop-produced state, and this note removes that loop. Nothing archived in this change; the
span-decoder and latent-z notes move to `rejected/` if G1–G3 of the spec read positive.

## Problem

Since 2026-09-04 the slot loop has been asked, through thirty-odd arms, to produce a vector
that helps the coda predict the next span. Every arm reads the same thing: token K1−K6
inside [−0.0001, +0.0033], passes 2–6 within 0.002 of pass 1, effective rank about 6 of
1024 across a row's slots, and a linear probe that scores the state at the loop's ENTRY
(AUC 0.60–0.64, null 0.51) no worse than at its exit.

Wolfe's 2026-09-14 reading, which this note adopts: the constraint that makes sampled depth
work (every depth reads out the same answer, so the loop is a fixed point) fights what z
needs to represent. Sharpened: a z trained only through "help predict what comes next" is
the expected thought over every continuation, and an expectation is reached in one pass,
whatever the loop is asked per pass. The two arms that put a target on the passes
(LoopMTP on the token loop, the horizon arm on the slot loop) confirmed it from the other
side: a per-iteration target defines a chain, the Poisson draw crushes a chain to a fixed
point, and a fixed-depth model is out of distribution at every other depth.

Three papers answer "what is z" the same way and it is the opposite of ours: the latent is
the code OF ITS OWN SPAN, defined by reconstruction with the encoder seeing that span
(LaDiR's VAE on one chain-of-thought sentence; LTM's z fit on the sequence by gradient
ascent), and PREDICTING the code is a separate job with its own loss and its own
iteration count (LaDiR's block-causal flow model, 5 → 30 steps worth +16 points).

## Proposal

The spec in full. In one paragraph: slot s's cells hold the code of span s+1. At training
time an encoder E makes the code from span s+1's own prelude states (M learned-query
pools, RMS-normalised, Gaussian noise added), and the coda, on the strict geometry, speaks
span s+1 reading that code plus the span's own token path. The core body is trained beside
this, one pass per slot, as a conditional-flow-matching velocity field from noise to the
(detached) code, conditioned on the tape of earlier codes and the slot's seed. At inference
E does not exist: the body samples the code in k Euler steps once per span, the coda speaks
per token with no core pass. Three phases in one run: define the code, learn to guess it,
roll out on guesses (LaDiR's load-bearing stage 2).

What the design keeps from the record: the strict geometry (the coda cannot bypass the
cells, measured at CE parity), the Bowman dropout, the boundary rule and packer, the CFM
math and its null-floor scaling from `fm_planner.py`, the rule that nothing iterated sits
in the CE graph, and the rule that no span is decoded from one vector without a token path.

## Alternatives considered

- **Keep the loop as the writer and change its target again** (a KL-regularised z, a
  contrastive target, more cells). Rejected: twelve targets read flat, and the probe says
  the state was already scoreable before the loop ran. The target was never the lever;
  the DEFINITION of z was.
- **The drawing board's blue variant: after z, one core pass per token with z as the entry
  state, then the coda.** Rejected for v0.1: it puts the thinker back in the per-token path
  (6 more block passes per token), shares one body between a velocity field and a token
  predictor (token CE would dominate it), and re-opens the credit path from token losses
  into z, which is the averaging that collapses z. Recorded as a follow-on if the mouth
  proves too small.
- **A separate planner network (FM1's shape) instead of the core body.** Rejected: the
  point is that the loop is the brain. FM1 also shows why a separate, detached planner
  writing into an additive prefix is unread. The body as the field costs no new blocks.
- **Regress z onto the code deterministically (JEPA / span-JEPA).** Rejected: a
  deterministic predictor of a code is the mean again, and the tree already forbids
  regressing onto the slot state (LCM T3/T4, CoCoMix §6b, BT §4.2). Flow matching is a
  regression of a VELOCITY toward a sample, and the sample is the thing a mean cannot be.
  The deterministic-predictor arm is nonetheless the right CONTROL for G3, and the spec
  lists it.
- **Token-level diffusion beside the head (dmorph) or DiffusionBlocks.** Rejected on the
  record: both measured and closed (`2026-08-30-objective-lines-vs-l2cap.md`,
  `dmorph-v1-design`).
- **A KL bottleneck (β-VAE) as the rate control.** Deferred to a second arm: LTM's
  amortised VAE collapsed with annealing; LaDiR's noise augmentation is one knob with no
  collapse dynamics of its own.
- **Diffusion-Forcing noise levels on the tape from the start.** Deferred to v0.2: the
  mechanism is exact but the decay law is a design choice the paper does not give, and it
  would be a second factor in the first panel.

## The binding rule

`2026-08-30-objective-lines-vs-l2cap.md`: "Any future FM-flavored proposal must target
post-core carriers or it is pre-refuted." What every P1-family design did: regress a
planner onto PRE-core pooled prelude features of the next span (pairwise cos 0.50,
eff-rank 40/1024, defined by nothing decodable), write the result detached into an
additive prefix the coda could bypass. The terminal reading was a planner
indistinguishable from noise on the best substrate, and the one co-trained arm that DID
retrieve the target (FM1, top-1 0.66–0.77) was unread by the coda at 0.0000 nats.

This design's target is a code defined by what the coda can decode, co-trained through the
coda, on a geometry where the coda must read the cells (0.1865 nats today, from a
deterministic write). That is the "what the reader needs" side of the boundary the rule
names, not the prelude-feature side. The note records this argument; whether it satisfies
the rule is Wolfe's call, and the first gate (G1: the code is a code, and it is read) is the
one that would have caught every P1 arm before a planner was trained.

**Wolfe, 2026-09-14: the FM binding rule is waived for TUL-Code.** The design proceeds; the
argument above stays as the record of why it is a different object from the P1 planners.

## Acceptance criteria

The spec's gates, frozen in the prereg before the run: G1 the code is a code (`ce_tf` ≥ 0.15
below the strict ruler at 5k, code rank ≥ 16); G2 the loop earns by sampling
(`ce_k1 − ce_k8 ≥ 0.02`, `fm/rel` < 0.9 in every band); G3 a sample beats a mean (`ce_k8`
below the strict ruler at 20k, CI excluding 0); G4 the ship bar (`ce_k8` within 0.02 of the
d1 rung at fewer block passes per token). C1–C10 of the spec pass on CPU before any GPU
run; a 12-step Spark smoke before the first queue entry; two seeds before any claim.

The note moves to `implemented/` when `tul_code.yaml` trains on master with C1–C10 green
and G1–G2 read, whatever they read. It moves to `rejected/` if G1 or G2 fails on two seeds.

## Risks

- LCM's setting (span-level latents on web text) lost to token AR; the token path, the
  coarse code and the measured 0.40-nat ceiling are the differences, and G3 can fail.
- The code moves while the field learns it (LaDiR freezes its VAE; we cannot from
  scratch). Detach on the target and the phase-2 delay; `fm/rel` is the check.
- `ce_k{K}` is a one-sample bound, not a likelihood; ship claims are partly generation
  claims and need the diversity guard.
- The first panel is one seed per arm; nothing in it is a verdict.

## Measured 2026-09-14 (panel at 5k, one seed each)

Filed in `lab/experiments/failures/2026-09-14-arc-tul-code.md`. The build holds C1–C10 and
runs at 3× the strict ruler's rate. E's code is a verbatim copy of the next span (the coda
reads it back at 0.4–0.6 nats; ten cuts show the sentence returned word for word), the
thinker's sample is close to an unconditional draw from the code distribution (residual
1.7–1.9× the code's variance in every principal subspace; newswire-shaped sentences with
no topic thread), and one sample costs the coda 0.68–0.71 nats against the ruler on
501k paired tokens. The K-curve is flat (k = 1 … 16 within 0.02). Phase 3 is load-bearing
(without it, +7.4). The flow ratio reaches 0.33 at 5k and is still falling, and the
code's content sits entirely in the top 128 of 1024 directions per cell, so the L2 flow
loss is not the defect: the target is. Acceptance criteria A2 and A3 fail; A1 and the
rate criterion hold. Status stays `proposed` pending Wolfe's call between (1) a frozen-E,
all-slots-rollout arm (`code_rollout_p: 1.0`, expected to return to the ruler) and (2) a
code defined by what the past determines rather than a copy of the span.

Addendum 2026-09-15: the K-sample marginal (8 draws) on the same checkpoint beats one draw
by 0.002 nats at k = 1 and 0.022 at k = 16 (a Jensen gap set by the spread of the draws; the log 8 cap is against the best draw), and moves −0.004 nats from
k = 1 to 16. The draws are near-interchangeable to the coda at 5k. Wolfe's call: read this
probe only on the full-phase 20k arms (the experiment file's addendum has the table).

Addendum 2026-09-15 (20k panel, one seed). The full phase set did not close the gap:
tul-code-20k one sampled draw sits +0.63 nats above the strict ruler twin on 501k paired
tokens at 20k (5k: +0.71). The sampler's Euler depth is worth −1.0 to −1.3 nats under the
8-draw marginal while the coda has only seen truth codes (phase 2, two seeds) and 0.00 after
5000 rollout steps. The thinker reaches its flow-loss floor by step 4000 and its sample stays
an unconditional draw at 20k (residual 1.7–1.9× the code variance; worth the top 32 of 1024
code directions). A defect found and fixed in the same change: a truth cell reached the
phase-3 coda at RMS sqrt(1 + noise²) and a sampled cell at RMS 1, eval fed bare z, and the
coda learned the norm as the sample flag (bare z 1.25 nats, the trained statistic 0.35 on the
same coda); the sampled gap is NOT that flag (a sample at the truth norm reads +0.01). Fix:
`tul.code_noise_renorm` (truth and sample share RMS 1) and eval/generation feed z at the
trained statistic. The resumed phase-3 arm `tul-code-renorm` tests whether the norm flag also
flattened the k-curve (`lab/experiments/planned/2026-09-15-tul-code-renorm-r10k.md`).

Addendum 2026-09-15 (thinker-only arm, `lab/experiments/failures/2026-09-15-tul-code-thinker-only.md`).
With E and the coda frozen at the parent's step 10000 and the thinker alone trained for
40k more steps at batch 12 on the fixed target, the flow share moves 0.320 → 0.279 and the
full-rank sample residual 1.9 → 1.72; the last 20k steps are worth 0.007 of share. The
frozen-target thinker learns slower per step than rollout1 did with the coda training beside
it. Training time is not the lever; the target is. The predictability arm (`code_target_lambda`)
runs next.
