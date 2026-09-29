# Agent Note: Per-depth persistent state in the looped core, and the "resonant depth" ideas parked beside it

Status: proposed

## Problem

The looped core carries ONE state between passes: pass t+1's first core layer reads pass t's
last core layer (plus the `DiagonalInjection` re-injection). Every layer inside the core sees
only what flowed up through the layers below it on the CURRENT pass. Layer l has no direct
access to what layer l itself computed on the previous pass.

Our measurements say the loop behaves like a power iteration: the first pass sets the scale
and the later passes rotate ([E7](../../../../lab/experiments/failures/2026-09-07-arc-e7-block-loop.md),
[iterative-map note](../../implemented/architecture/2026-06-19-iterative-map-dynamics.md)).
A single carried state is one reason a loop can do that: each pass can only re-read one vector
per position. A carry that keeps one state per depth gives every layer its own memory across
passes, which a single carried stream cannot.

Source: Wolfe's 2026-09-29 discussion of a "resonant Transformer" (up sweep, down sweep, up
again, with persistent per-layer states `h_0..h_L`). Wolfe: "We may want to test per depth
state. Save a note about the rest for later testing."

## Proposal

**Test first: per-depth persistent state.** Inside the looped core, layer l on pass t+1 reads
its own output from pass t through a gated add at its input:

    x_l^{t+1} <- x_l^{t+1} + g_l * P_l(h_l^t)

- `P_l`: a d x d projection per core layer, zero-init, so step 0 IS the current model.
- `g_l`: a per-layer scalar or per-channel gate. No learned noise scale.
- Pass 1 reads nothing (no previous pass); the Poisson depth draw and full BPTT are unchanged.
- The state is causal: it is per position, like the residual. No future token enters.

Smallest falsifiable run: the plain looped model (`notul` recipe, norm_match ternary, 1000-step
ramp, seq 1024, batch 6), 5000 steps, one arm with the per-depth read and the unchanged
control. Readouts: K1-K6 on the 480 sweep rows, depth-6 CE paired on the same rows, the pass-0
write (`loop/core_gain_t0`), and the per-layer gate values. Memory: L x (B, T, d) extra
activations per pass under full BPTT; trace it before queueing.

If it earns on the plain loop, port it to the slot loop's `_tul_core` (per slot, per cell).

**Parked for later, with what our data already says about each:**

1. **A learned top-down pass** (`g_L .. g_1` after `f_1 .. f_L`, then up again). U o D is still
   one looped map with a bottleneck at h_0; per E7 it may still act as a power iteration. Worth
   a test only if the per-depth state earns and a directional split is the next question.
2. **Decoding from h_0** (the embedding side) instead of h_L, or both with deep supervision.
   With re-injection, h_0 is anchored to the CURRENT token, and the LM head is weight-tied to
   the embedding (`lm_weight()`), so the head asks h_0 for the NEXT token too: two jobs for one
   state. Expect the down path to learn to copy. Needs a detach or a separate head, and a
   check for the identity escape.
3. **Energy / equilibrium settling** (predictive coding, DEQ: update H to minimise
   `sum ||h_l - U_l(h_{l-1})||^2 + beta ||h_{l-1} - D_l(h_l)||^2`). This is a fixed-point
   objective. The fixed-point term at 1.0 held the slot loop still and killed depth use; 0.1
   or off earns ([fixed-point note](../../implemented/architecture/2026-09-07-fixed-point-objective-as-the-loop-stability-term.md),
   [fp01 10k](../../../../lab/experiments/successes/2026-09-25-lxtul-fp01-10k.md)), and every
   per-pass target we tried was met in one step. It is also forcing, which Wolfe ruled out
   (depth use is emergent). Do not run it as the loss.
4. **Modes with |lambda| near 1 ("resonance")**. The map drifting to gain 1 is the slot loop's
   spike train and the detonation cliff steepens like rho^T
   ([break glass](../../../../lab/divergence/BREAK-GLASS-IN-CASE-OF-DIVERGENCE-THE-SLOT-LOOP-GAIN-CONSTRAINT.md)).
   Not a target to aim at; a region our constraints already guard.

## Alternatives considered

- **Widen the diagonal carry** (Parcae's rho(A) < 1 on the whole residual): run as
  [E9](../../../../lab/experiments/failures/2026-09-07-arc-e9-widen-the-carry.md), a stability
  test, failed. It still carries ONE state; it does not give layer l its own past.
- **A carry beside the slot read** (the sum / gate carry of 2026-09-20, the LX carry stage 0 of
  2026-09-26): the carry replaced the read and diluted it. That carried the slot's EXIT state
  between passes, not a per-layer state. The per-depth read adds to, never replaces, the
  current pass's input, and starts at zero.
- **GLA, the gated linear-attention retention branch** (Wolfe, 2026-09-29: "we did try GLA and it
  was problematic"). GLA carries a recurrent state across TOKENS: the retention carry was a
  learned causal leak (0.14 nats truncated, 3.85 after 30k), the naive write-alignment retrofit
  failed ([2026-08-31](../../../../lab/experiments/failures/2026-08-31-gla-write-alignment.md)),
  it was cut from the winner recipe when the conjunction killed depth-earning, and under the LR
  ramp it was neutral on every axis ([2026-09-02](../../../../lab/experiments/failures/2026-09-02-a2-gla-under-warmup.md)).
  The per-depth state recurs across PASSES at one position: it never reads another token, so it
  has no leak path, and it is a residual read, not a second attention branch.
- **The full resonant architecture at once** (up/down sweeps + per-layer states + h_0 decode):
  rejected as a first test; three changes in one arm cannot say which one earned.

## Acceptance criteria

- The per-depth arm's K1-K6 beats the control's by more than the control's seed spread (run
  the control at two seeds; the LX-Fan family showed single draws cannot resolve K1-K6).
- Depth-6 CE is not worse than the control by more than 0.0137 on the same rows.
- No detonation; the pass-0 write reported beside the control's.
- The gates `g_l` move off zero (if they stay at zero the arm is the control and says nothing).

## Risks

- The per-depth read can turn the loop into a copy of the previous pass (h_l^{t+1} ~ h_l^t),
  which looks like convergence and earns nothing. The gate values and K1-K6 catch it.
- Memory under full BPTT: one extra (B, T, d) state per core layer per pass.
- The prior notes above were measured on the slot loop and the plain loop at 5k-10k; none of
  them tested a per-layer carry, so they bound the parked ideas, not this one.
- The literature trail offered for this idea (a 2026 predictive-coding survey, "Loop, Think, &
  Generalize" 2026) was not verified. Huginn (3.5B, 800B tokens) and Racecar training (Xie and
  Thuerey) are known.
