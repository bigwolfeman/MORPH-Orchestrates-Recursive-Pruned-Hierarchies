# Agent Note: loop stability mechanisms from the literature, and the order to test them

Status: proposed

## Problem

MORPH's looped core diverges in three measured regimes and every cure in the tree is a
recipe (a 1000-step LR ramp, an optimizer cap) or a penalty that can lose (the finite-
difference gain hinge lost at a deep slot draw: `lab/experiments/failures/2026-09-07-arc-
e7-block-loop.md`). E7's Jacobian names the mechanism: one iteration sets the carrier's
scale (485 → 1.5e7), the rest rotate, the state collapses to rank 9, and the backward through
a block with σ_max 9e4 detonates. A 2026-09-07 re-read of Parcae and a survey of the
Awesome-Loop-Models list (`docs/references/looping-depth/parcae/2026-09-07-parcae-vs-morph-
mechanisms.md`, `docs/references/looping-depth/2026-09-07-loop-stability-survey.md`) give
six mechanisms with measured evidence that target that picture rather than its symptoms.
Wolfe, 2026-09-07: "save a note about it, try the top 3 or 4."

## Proposal

Test each mechanism under ONE bar first: the measured detonation assay (`notul`, warmup 0,
1200 steps, `preclip/total > 1e4` at step ≥ 200 = detonation, ~70 % base rate;
`lab/divergence/DIVERGENCE-README.md`), six draws per arm and three concurrent control
draws, at most one detonation in six to call it. A mechanism that passes goes to a 5000-
step ramped run against notul (val CE and the depth sweep) before it touches base.yaml.
In order:

1. **Widen the diagonal carry to the whole residual** (Parcae §4.1). `model.injection_
   channels: all`. Built and armed as E9 (`planned/2026-09-07-arc-e9-widen-the-carry.md`).
2. **A terminal fixed-point objective** (2608.18222). `model.core_fixed_point_lambda`:
   λ · mean over samples of ‖h_T − h_{T−1}‖² / ‖h_T‖² at each sample's LAST iteration. It
   asks the loop to settle, which targets the drift to gain 1 and the "earns nothing past
   3" reading with one term. E10a.
3. **A directional gain hinge on the plain loop** (STARS 2605.26733, done by finite
   difference instead of a JVP). `model.core_gain_lambda / core_gain_target /
   core_gain_direction: power`: the probe direction is a persistent buffer updated by one
   power step per training step (v ← normalize(f(h+d) − f(h))), so the hinge reads the
   map's TOP singular direction instead of a random one. E7's checkpoint read 0.85 on the
   random direction while σ_max sat at 95; healthy arms read 11–22 there, so the target
   is on σ_max (20), not on the typical gain. E10b.
4. **DeepLoop residual scaling for weight-tied depth** (2607.13491, 2606.18524): scale
   the shared block's write into the carrier by the unrolled depth. MORPH's residual is
   the Cayley hyper-connection, so the write scale is its β/read-write vectors, not a
   plain residual constant; needs a reading of `hyper_connections.py` before it is a knob.
   Next window.
5. **A softmax-gated write** (Fully-Looped Transformer, 2605.18797): the loop's write
   into the carrier goes through attention (Q = the carried state, K/V = the block
   output) so it is bounded by construction. An architecture change to the core block;
   next window, only if 1–3 do not hold.
6. **A per-sample gradient window** under truncation (Parcae Alg. 2): a code defect, not
   a mechanism; `.agents/notes/proposed/bug-fix/2026-09-07-truncated-bptt-silences-shallow-
   samples.md`. Must land before any deep-draw arm is read again.

## Alternatives considered

- Another spectral or Lipschitz cap on the core weights. Refuted four times in this tree
  and now explained: LayerNorm makes the recurrence Jacobian non-normal, so an operator-
  norm bound is the wrong budget (2607.10681), and no spectral regime alone gives input
  dependence without recall plus outer normalization (2604.15259).
- Raising the hinge's λ or lowering its target. E7 shows the random-direction hinge is
  blind to a rank-few expansion; more of it cannot see more.
- A lower LR (Huginn cut 10x; Parcae's baselines fail above 4e-4). MORPH already trains
  20–80x below Parcae's unstable baselines; the ramp is the same symptom control.
- The bounded-write mechanism first. It is the strongest bound but the largest change;
  the carry and the two loss terms are config-level and share one assay.

## Acceptance criteria

- Each arm's prereg frozen before its smoke; verdicts by the abort rule per draw.
- A mechanism is "held" only if ≤ 1 of 6 warmup-0 draws detonates AND the concurrent
  control detonates ≥ 2 of 3.
- Bit-identical default for every knob (tests), full suite green.
- Whatever holds is then measured at 5000 ramped steps against notul on the 480 rows
  (CE at the trained depth and the depth sweep) before any default changes.

## Risks

The assay is a warmup-0 proxy for the early detonation; the slot-loop spike train and the
deep-draw power-iteration failure are later and different, and a mechanism can pass the
assay and still lose there (E7's hinge passed every assay it was given). The fixed-point
term can be satisfied by a trivially contractive map that earns nothing (E1's lesson:
stability dials are not earning dials). The finite-difference power step is a one-step
approximation of the top direction and lags a fast-rotating map.
