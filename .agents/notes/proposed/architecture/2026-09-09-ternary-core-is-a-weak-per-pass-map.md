# Agent Note: the ternary core is a weak per-pass map, and that is why the loop earns nothing past pass 2

Status: proposed

## Problem

Two panels on 2026-09-09 closed every other candidate for why MORPH's looped core converges
by pass 3 and earns 0.03 nats where Parcae's earns 0.29:

- The Parcae-entry panel (`lab/experiments/failures/2026-09-09-arc-e19-parcae-loop-entry.md`)
  rebuilt the loop's entry the way Parcae does it: state from noise, e re-injected on every
  dimension through a learned B, no fixed-point term. It gave a 0.023-nat better model with a
  start-independent fixed point, and the loop still earned 0.033 past pass 1 and 0.0009 past
  pass 3. Its depth-1 control sits within 0.02 nats of the looped plain model at depth 6 at 5,000
  steps; that is a short-horizon reading (deep models converge slower), not a ranking.
- The depth-candidates panel (`lab/experiments/failures/2026-09-09-arc-e20-loop-depth-candidates.md`)
  ran Parcae's depth schedule (no loop contribution: K3−K8 0.002), a carry that trains at 20×
  the rate (B diagonal 1.74, the carrier collapses to rank 6, K3−K6 0.007), and a
  diagnostic arm with ternary everywhere except the looped core. Only that arm moved the loop's
  contribution: its value past pass 1 went 0.033 → 0.168 and past pass 3 went 0.0009 → 0.012
  (the CE gaps at 5,000 steps are not a ranking of ternary against bf16; ternary is the
  recipe and this arm is a diagnostic).

The anatomy names the mechanism. At iteration 6 the bf16 core's MLP branches emit 0.75–0.96
of their input and its attention 0.14–0.43; the ternary core's MLPs emit 0.19–0.24 and its
attention 0.08–0.21. Its state moves 63/38/20/12/9/6 % per pass against the ternary core's
37/23/7/5/3/2 %. A pass through the ternary core barely changes the state, so the state
sits on its fixed point by pass 3 and the coda has nothing more to read. Under the
symmetric ternary scale (γ = mean|W|, threshold 0.5) each quantized layer's output is
smaller than its bf16 twin's, and a gate-up-down MLP compounds three of them per block.

Ternary is not negotiable (Wolfe's standing rule: no dense-then-ternary; ternary weights
organize differently). So the question is how a ternary core can be a strong per-pass map.

## Proposal

Restore the per-pass branch strength under ternary and measure it with the three standing
instruments (the K-curve first; token-paired CE against `parcae-entry` at depth 6 and the
depth-1 control reported as 5,000-step readings), one factor per arm on the Parcae-entry recipe:

1. **Learnable ternary scales on the core's MLPs** (`ternary_scale_mode: ttq`, which the tree
   already has: one learnable γ₊ and γ₋ per group, initialised from mean|W|), scoped to the
   core so the prelude and coda stay on the shipped symmetric mode. The hypothesis is that
   the optimizer grows γ where the loop needs a stronger map. The anatomy's branch ratio at
   iteration 6 is the direct readout: the arm passes its mechanism check if the core MLPs
   read above 0.5.
2. **Norm-matching scale** as the control on the same idea without new parameters: γ chosen
   so ‖γ·codes‖ = ‖W‖ per group instead of γ = mean|W|. Needs a new scale mode; a
   one-line change in the grouped STE with a test against the symmetric mode.
3. **Finer scale groups on the core only** (`ternary_scale_group: 128` against the shipped
   per-tensor scale) — the cheapest arm; a per-row scale lets rows with large weights keep
   their magnitude.

Every arm is scored at 5,000 steps with `core_depth_sweep`, `core_anatomy` and
`core_init_probe`, the branch ratio being the mechanism instrument and K3−K6 > 0.02 with
the CI above 0 the depth bar. The density panel (`lab/experiments/planned/2026-09-09-arc-density-panel.md`)
runs first and says whether the production prune moves the same map the same way.

## Alternatives considered

- **A dense (bf16) core in production.** Rejected: it is the one arm that moved loop
  contribution, but ternary is the recipe the project exists for, Wolfe's rule forbids a
  dense-then-ternary path, and a 5,000-step CE gap ranks nothing. It stays a diagnostic.
- **Muon or a higher base learning rate for the core.** Deferred: the carry-rate arm
  showed that a faster carry alone collapses the state's rank without adding contribution; a
  core-wide rate change is a second factor on top of an unresolved first one.
- **Parcae's depth schedule.** Rejected by measurement (K3−K8 0.002; no loop contribution).
- **More blocks per pass or fewer streams.** Not tested; it changes the architecture the
  checkpoints depend on and does not address the measured mechanism (branch magnitude).
- **A readout probe first** (a linear head fit per iteration). Kept behind the ternary
  lever: the mechanism is now measured at the branch, so the cheaper test is to fix the
  branch and re-read the K-curve.

## Acceptance criteria

- One arm whose core MLP branch out/in at iteration 6 reads above 0.5 (anatomy, 3 rows).
- That arm's K3−K6 > 0.02 with the CI above 0, or its K1−K6 > 0.10, at 5,000 steps.
- Its token-paired CE at depth 6 against `plain-depth1` at depth 1 and against `parcae-entry`
  at depth 6 is reported as a 5,000-step reading (not a ranking: the break-even for a deeper
  map lies at a longer horizon).
- The arm stays ternary end to end: the deploy quantizer reads the same codes; no bf16
  weight survives in the core at export.

## Risks

- A larger γ under STE may re-open the detonation the 1000-step ramp cured; the sustained
  tripwire and the divergence README's table apply.
- The bf16 diagnostic converged by pass 6 too (K3−K6 0.012, not Parcae's 0.040 at K3−K8);
  the ternary lever can at best reach that, and the remaining gap (block and stream count,
  optimizer, tokens) is a separate question.
- Three rows of anatomy is a small sample for the branch ratio; the readout is a ratio of
  norms and was consistent across the eight arms measured, but a training-wide log of the
  same ratio would be the better instrument.
