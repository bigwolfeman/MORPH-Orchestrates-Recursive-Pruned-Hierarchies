# Agent Note: the slot loop's contraction rate is a design decision, not a symptom

Status: proposed

Date: 2026-09-04. Branch `tul/think-once`. Trigger: Wolfe, 2026-09-03 — "we haven't
considered whether we are contractive or not in the first place and what the implications
of that are", "the generalization of the map is the most important aspect of the looped
core", "I feel like the loop wants to be asymptotic to 1 and sometimes we are stepping
over the edge". Evidence: `lab/experiments/successes/2026-09-03-tul-onset-capture.md`
and `lab/divergence/DIVERGENCE-README.md` §D.

## Problem

MORPH runs one weight-shared core over T iterations with a per-sample Poisson depth. That
design only makes sense in ONE regime of the iterated map `h_{k+1} = f_θ(h_k)`: a slow
contraction toward a fixed point, where the state keeps moving for every extra iteration
and a deeper draw refines the same answer. Nothing in the tree chooses that regime. The
loss sees only `∇_θ L`, which integrates over the trajectory, so the optimizer is blind to
the map's gain (CLAUDE.md, "nested dynamical system"). The onset capture measured where
the slot loop actually sits on the forecast target:

- Typical one-step gain `jac/rms_t3` drifts from 0.87 to 1.00 over 1200 steps with the
  loss (Spearman 0.98 against step). The spike train begins within 0.05 of 1.
- The map is anisotropic: each block's typical gain is 1.00–1.02 and the whole map
  contracts a generic direction by ~0.9 while its worst direction is amplified 100x by
  the onset (3x at init). The amplified directions are aligned across the six shared
  blocks.
- The trajectory is NOT near a fixed point: the relative step size per iteration stays at
  0.5–0.9 through all eight iterations and the successive-step ratio sits at 0.7–0.78
  (0.75–0.87 on spike steps). The loop is doing sequential computation, not refinement.
- Ternary QAT is necessary for the spike train (ternary off: clean), but the impulse is
  not a code-flip burst; the spike is the backward product of eight iterations through a
  map whose gain has reached 1.

So "asymptotic to 1" is the measured behaviour, and the failure is the side effect of
letting the target pull the gain there with nothing to hold it. Three regimes and what
each implies for the architecture:

| regime | one-step gain along the trajectory | what a deeper draw does | fits Poisson depth? | toolkit |
|---|---|---|---|---|
| slow contraction | 0.7–0.95, step size shrinking geometrically | refines the same answer; K1..K8 keeps improving and saturates late | yes | a rate held below 1 by construction; a fixed-point loss |
| fast contraction | < 0.5 | nothing after 2–3 iterations (K4 saturation, seen on every token-loop arm) | wasted depth | none needed; the loop is shallow |
| marginal / expansive | ≥ 1 along some direction | sequential steps; the backward product grows as gain^T | no: depth is a program length, not a refinement count | RNN toolkit: clip-through-time, gating, bounded state |

## Proposal

1. **Measure the regime as a first-class training metric**, every N steps, on every looped
   arm: `jac/rms_t{k}` (typical gain), `jac/sigma_t{k}` (worst gain), and the
   successive-step ratio `loop/delta_ratio_t{k}` along the trajectory. The instruments
   exist since the capture; the proposal is to put them in `base.yaml` at a cadence that
   costs < 2 % of the step and to log the gain's drift as a wandb panel next to the loss.
2. **Decide the regime.** For the slot loop the intended regime is slow contraction (the
   thought is refined, and a deeper draw at inference keeps helping). Hold the rate there
   by construction on the MAP, not on one block's weights: candidates in order are
   (a) a direction-preserving renormalization of the carried state between iterations
   (bounds the state, leaves the operator alone), (b) a fixed-point residual loss
   `‖h_{T} − h_{T−1}‖` with a small weight (pulls the trajectory toward a fixed point;
   the record's Fixed-Point Forcing on dmorph failed for a different reason — one answer
   per position was the wrong target there), (c) clip-through-time on the cotangent
   between iterations (bounds the product the spikes are made of; forward untouched).
   Each is one prereg on R4 with the tripwire and the gain panel as readouts.
3. **Retire the weight-spectrum levers on the loop for good.** Four caps failed and the
   capture shows why: the per-block gains never left 1.00–1.02 while the map's worst gain
   went 3 → 500. A uniform bound on one factor cannot see the alignment of six.
4. **Say what space the loop refines.** The loop acts on the flat residual stream (the
   Cayley hyper-connection mixing is an isometry); the hyperbolic embedding shapes the
   INPUT to the loop and nothing inside it. A fixed-point reading needs a metric on the
   stream; the effective rank of the slot states (falls 172 → 48 across the eight
   iterations on calm steps) says the trajectory collapses onto a low-dimensional set
   even while the step size stays large. Whether that set is the attractor we want, or a
   collapse, is the question the depth sweep on a surviving arm answers.

### Status 2026-09-04 10:00

Candidate (c), clip-through-time, is built (`model.slot_cot_clip`, `4d5a986`) and measured
on M-next: the clip bound on every step from 1736, the exit cotangent stayed flat, and the
run tripped at 2764 with the forward inflating (iteration 0's realised gain 1.5 → 9.6,
exit norm 5x, weight gradients 800x). Bounding the backward alone does not decide the
regime. Candidates (a) and a finite-difference typical-gain penalty (a trainable version
of "hold the rate below 1 on the map") are built (`34d94a0`) and pre-registered
(`lab/experiments/planned/2026-09-04-tul-forward-levers.md`).

### Status 2026-09-04 12:20

Both forward levers measured (`lab/experiments/successes/2026-09-04-tul-forward-levers.md`):
M-next survives 5000 steps under either, with zero spike steps. The gain penalty holds the
fp32-measured typical gain at 0.887–0.897 for the whole run — the first acceptance
criterion is MET (below 0.95 throughout). The second is NOT: the stable loops' step ratio
falls with the iteration index (0.3–0.5, refinement-shaped) but forecast K3−K6 is 0.000
and the coda reads nothing. The third (log the gain on every looped run in `base.yaml`)
is open. Decision left to Wolfe: which lever ships, and whether the next arm is the
deep slot stack or the "what is a deeper draw for" question.

## Alternatives considered

- **Keep clamping the symptom** (`core_gain_clip`, spectral caps): refuted four times on
  the takeover and once more here; the forward norm is flat on every spike step.
- **Drop weight sharing on the slot path** (Wolfe, 2026-09-03: "I'm hesitant to touch it
  here"): removes the alignment across blocks by construction but also removes the loop;
  parked, not rejected — it is the deep-slot-stack arm of the think-once decision rule.
- **Treat the loop as an RNN and ship clip-through-time alone**: the cheapest fix for the
  spikes and it leaves the regime undecided; the gain would keep drifting and the loop
  would keep doing sequential computation under a Poisson depth that assumes refinement.
  Kept as candidate (c), not as the design.
- **Ternary hysteresis first**: the section-A lever; no evidence on this face, the flip
  burst is absent. Not first.

## Acceptance criteria

- A forecast arm (R4 recipe) reaches 5000 steps with the tripwire silent under one of the
  levers in (2), and its `jac/rms_t3` stays below 0.95 for the whole run (prereg with the
  number frozen before launch).
- On that arm the successive-step ratio falls with the iteration index (refinement), and
  the slot depth sweep's K1−K6 CI sits above 0.
- `jac/rms_*` and `loop/delta_ratio_*` are logged by `base.yaml` on every looped run.

## Risks

- A rate held below 1 may hold the loop's earning down with it (the ramp cut the plain
  model's loop earning from 0.207 to 0.04 nats by keeping the map near identity). The
  target is a rate, not identity; the prereg must score the depth sweep, not the loss.
- The typical gain is a Frobenius reading over the active slot positions on one batch.
  The sweep on the saved checkpoints (`jac_sweep.jsonl`, one fixed batch) reproduces the
  in-run drift to within 0.01 and adds: ternary off holds it at 0.89; M-own with ternary
  moves 0.90 → 0.91; A1 (bptt 4, token CE) sits at 0.975 with a worst gain of 176 and
  NO backward growth. The gain drifting to 1 is the condition, the backward product is
  the event; a lever must be scored on the product (`loop/cot_norm_*`) as well.
- The paid (token) loop sits ABOVE 1 in typical gain while healthy (1.05 → 1.13, A2
  ladder). "Gain at 1" is not the detonation by itself; the spike train needs the forecast
  target's pull AND ternary. This note is about the slot loop.
