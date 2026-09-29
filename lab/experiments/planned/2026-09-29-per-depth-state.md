# Planned: per-depth persistent state in the plain looped core (2 seeds, paired controls)

Status: planned

Date: 2026-09-29 03:55 (frozen before any of the four runs trained).
Design: [`../../../.agents/notes/proposed/architecture/2026-09-29-per-depth-state-and-resonant-depth.md`](../../../.agents/notes/proposed/architecture/2026-09-29-per-depth-state-and-resonant-depth.md).
Wolfe (2026-09-29): "Per depth state may be different enough though. Have a subagent build and add
it to queue."

## Question

The looped core carries one state between passes. If each core layer also reads its OWN output
from the previous pass (`model.core_depth_state`: zero-init projection P_l, per-channel gate
g_l, added at the layer's input from pass 2 on), does the loop earn more depth use?

## Hypothesis

H: a per-depth read gives each layer a memory across passes that the single carried stream lacks,
so later passes do more work: K1-K6 and K3-K6 rise over the control at both seeds.

Counter-hypothesis I hold about as strongly: the recipe's fixed-point term (1.0) rewards a small
last step, and a layer that can read its own previous output can meet it by copying. Then the
gates grow, depth use does not, and K3-K6 may fall.

## Predictions (mine, orchestrator, before any run)

Reference: plain-panel-norm-match at 5k on older code, K1-K6 +0.0528 [+0.0513, +0.0543], K3-K6
+0.0088. Readouts on the 480 sweep rows.

- **P-1**: neither per-depth run detonates. 85 %.
- **P-2**: the control rerun on current code (s1r) reads K1-K6 within 0.010 of +0.0528. 70 %.
- **P-3**: per-depth K1-K6 exceeds its same-seed control's at BOTH seeds. 40 %.
- **P-4**: per-depth K3-K6 exceeds its same-seed control's at both seeds. 40 %.
- **P-5**: per-depth depth-6 CE is within 0.0137 of its same-seed control's, paired on the same
  rows, at both seeds. 70 %.
- **P-6**: per-depth training tok/s is at least 90 % of the control's. 80 %.
- **P-7**: the gates' mean |g_l * P_l| read at 5k is largest on the LAST core layer. A
  diagnostic, not a pass line. 50 %.

Pass: P-3 and P-5 hold.

## Method

- Code: `model.core_depth_state` in `_core_region` / `_apply_core_step` (morph/model/transformer.py),
  wired through `build_morph_config`, default false in base.yaml. Off is bit-identical to the
  tree before the change (test against a `git archive` of b4753d2). Refuses SCSE, core_impl
  parcae and core_gain_lambda > 0 (the gain hinge would probe a map without the read).
  Tests: `tests/test_core_depth_state.py`.
- Configs: `notul_panel_norm_match_ds` / `_ds_s2` (knob on, seeds 1 / 2) and
  `notul_panel_norm_match_s1r` / `_s2` (controls on current code, seeds 1 / 2). Recipe:
  plain panel, norm_match ternary, 1000-step ramp, seq 1024, batch 6, Poisson(6) depth clamped
  to [1, 8], full BPTT, fixed-point term 1.0, 5000 steps. Composed: each differs from
  `notul_panel_norm_match` only in the knob, the seed and wandb.name.
- Order: ds s1, ctrl s1r, ds s2, ctrl s2, queued after the epivol seed bands and before grid 2's
  held arms. Runner `runner_steps.sh`, guard `perdepth_guard.sh`.
- Readouts: depth sweep 1-6 (and 9 to 16) on 480 rows; paired depth-6 CE ds vs control per seed;
  the gate and projection norms per layer from the checkpoint. The runner's TUL-only readouts
  (stage-2 scorer, worth) are not readouts of this experiment.
