# Agent Note: the slot-loop gain hinge reads measurement noise at `slot_gain_eps` 0.02 on the eager path

Status: proposed

Date: 2026-09-08. Found by the Olympiad throughput audit
(`lab/perf/2026-09-08-oly-throughput-audit/`), not by a stability run.

## Problem

The slot-loop gain constraint (`model.slot_gain_lambda` 100 at target 0.9,
`.agents/notes/implemented/architecture/2026-09-04-slot-loop-gain-constraint.md`) estimates
the map's directional gain `g` by a finite difference of relative size `slot_gain_eps`
(0.02 in `base.yaml`) at one random grad iteration per step, and pays a hinge on
`g − 0.9`. A finite difference of a bf16 map is a norm ratio with noise in the numerator:
`g_measured² ≈ g_true² + (noise / eps)²`. The bias is upward and shrinks as `eps` grows.

Measured on `tul_oly_mask` at fresh init, the same weights and the same random
directions in both modes (`scripts/gain_eps.py`; eager = `use_kernels false` process-wide,
fused = `tg_scoped_kernels true`):

| `slot_gain_eps` | eager mean | eager sd | fused mean | fused sd |
|---|---|---|---|---|
| 0.02 (shipped) | 0.9416 | 0.0139 | 0.8717 | 0.0022 |
| 0.05 | 0.8920 | 0.0040 | 0.8704 | 0.0012 |
| 0.10 | 0.8795 | 0.0019 | 0.8701 | 0.0010 |
| 0.20 | 0.8750 | 0.0011 | 0.8700 | 0.0010 |
| 0.40 | 0.8734 | 0.0010 | 0.8698 | 0.0010 |

Both paths converge on 0.870. At the shipped eps the eager path reads the map 0.07 high
with 7x the spread, so on every eager TG arm (E4, E13's mask arm, E14) the hinge at
target 0.9 was firing on noise while the map sat at 0.87. The fused path reads the map
at every eps. The eager path's extra noise is the eager attention's own bf16 rounding
(the fused kernels accumulate differently); it is not a bug in the estimator's algebra.

What this changes about the record: the E13 mask arm's hinge penalty was not zero
(`loss/gain_reg_weighted` > 0) for a map that was under target. Its verdicts (share, not
depth) do not depend on that: the K-diffs are measured on the trained model, and the E14
expansive dial moved the TARGET, which the bias shifts by a constant. The hinge's role
as a HOLD (the spike train at gain 1) is unaffected in kind, since a map drifting to 1.0
reads ≥ 1.0 in both modes; but its onset on an eager arm was 0.07 early.

## Proposal

1. Every TG arm runs `model.tg_scoped_kernels: true` (E16 does; `tul_oly_mask.yaml`). The
   panel's four arms then read the hinge the same way.
2. Raise `slot_gain_eps` to 0.1 in `base.yaml` so the reading is honest on either path.
   At 0.1 the eager bias is +0.009 and the fused reading moves by −0.0016 (a no-op). NOT
   done in this change: it changes the constraint on every slot-loop arm and needs one
   paired run (mask at eps 0.02 vs 0.1, fused) before it ships, filed as an arc experiment.
3. Instrument: log `loss/gain_est` next to `loss/gain_est_max` on every arm (already done)
   and record in the divergence README that an eager reading at eps 0.02 carries +0.07.

## Alternatives considered

- **Re-derive `slot_gain_target` for the eager path (0.97).** Rejected: it bakes a
  path-dependent constant into a config and the eager path is the slow one nobody
  should run.
- **A central difference or a two-eps Richardson estimate.** Halves the bias per eps but
  costs a third core application per step (the hinge is already 6 % of the forward).
  Rejected for cost; raising eps buys the same for free.
- **Leave it.** The hinge's job is the spike train, and at gain 1 both paths agree.
  Rejected: a constraint that fires when its instrument is wrong is a constraint whose
  price (E14: 7.9 % of the step) is paid for nothing.

## Acceptance criteria

- On the fused path the mask arm's `loss/gain_est` mean over steps 1000–6000 sits under
  0.9 with a hinge penalty on fewer than 5 % of steps (E16 P16i).
- A paired mask run at eps 0.1 vs 0.02 (fused) ends within the E13 seed spread on token
  CE and K1−K6, with `loss/gain_est` sd halved.

## Risks

- A wider eps measures a less LOCAL gain; at 0.1 the perturbation is 10 % of the state
  norm. The sweep shows the reading is flat from 0.1 to 0.4, so the map is near-linear at
  that scale, but that was measured at init only.
- The E13/E14 records cite hinge readings; they stay as recorded, with this note linked.
