# BREAK GLASS IN CASE OF DIVERGENCE: the slot-loop gain constraint

Shipped 2026-09-04. Read this when a looped-core run spikes and detonates after the LR
ramp ends. It is short on purpose; the long records are linked at the end.

## 1. What to do

The constraint is ON in `morph/configs/base.yaml` for every model that has a slot loop:

```yaml
model:
  slot_gain_lambda: 100.0   # THE constraint: a hinge on the slot map's typical gain
  slot_gain_target: 0.9     # the gain it holds the map at
  slot_gain_eps: 0.02       # finite-difference step, relative to each slot's norm
  slot_cot_clip: 4.0        # the belt the measured configuration wore (per-row cotangent cap)
  slot_state_renorm: false  # the other measured lever; either one, not both, was tested
```

If a slot-loop run still shows the spike train, check three things in order:

1. **Is the constraint on the path?** The model prints `[slot-levers] {...} INERT on this
   model: <why>` at build when it has no slot loop (no TUL block, `n_core: 0`, the paid
   loop's `tul.tokens_through_core`, or an FM planner). The levers act only inside
   `_tul_core`. A run that printed INERT is not protected by this file.
2. **Is the penalty binding?** `probe.jsonl` (with `training.grad_probe_every=1`) carries
   `loss/gain_est` (the live typical gain), `loss/gain_est_max` and
   `loss/gain_reg_weighted`. Healthy: `gain_est` 0.88–0.90, penalty touching 15–35 % of
   steps with a median of 0.000 (a barrier, not a tax). A `gain_est` that sits above the
   target for hundreds of steps means lambda is too small for the shape; double it.
3. **Is the drift somewhere the penalty does not look?** The penalty regularises ONE random
   grad iteration per step. Iteration 0 (the entry state) is drawn 1 time in 8 and its gain
   still drifts (0.94 → 0.996 over 5000 steps on the measured arm). If a run detonates with
   `gain_est` held, read `jac/rms_t0` from `lab/divergence/jac_sweep.py` on its checkpoints
   before changing anything else.

Turning it off (`slot_gain_lambda: 0.0`, `slot_cot_clip: 0.0`) is bit-identical to the
tree before it existed.

## 2. What it is, in five lines

A slot loop trained through all its iterations (full BPTT) with ternary QAT drifts the
typical gain of its map `f_θ` from 0.87 at init toward 1.00 once the LR ramp ends (Spearman
0.98 against step). The spikes begin within 0.05 of 1 and the run detonates at the
crossing: the backward product through the eight iterations reaches 39–2436x on spike
steps while the forward stays flat. The penalty `lambda · relu(g − target)²` measures `g`
every step by `‖f(h + d) − f(h)‖ / ‖d‖` at one iteration (two extra core steps on the
compact slot sequence, same dropout masks, the global RNG put back) and holds it at the
target. It acts on the whole map, which is the quantity that drifted.

## 3. What was measured (2026-09-04, seq 1024, batch 6, winner recipe + 1000-step ramp)

| arm | constraint | outcome | fp32 typical gain at iteration 3 over the run |
|---|---|---|---|
| M-next, fused (panel) | none | detonated 3618 | — |
| M-next, eager, deterministic | none | detonated 1208 | 0.885 → 0.914 (1150) → 1.000 (1200) |
| M-next, ternary OFF | none | clean to 5000 | 0.889 flat |
| A1 at `bptt_depth` 8, two seeds | none | detonated 1682, 1144 | 0.89–0.91, iteration 0 at 0.999 before the onset |
| M-next + `slot_cot_clip` 4 alone | backward only | detonated 2764; forward inflated 5x; gain 0.89 → 8.7 | 0.894 (1500) → 1.885 (2000) → 8.663 (2500) |
| M-next + clip + `slot_state_renorm` | forward | clean to 5000, 0 spikes | 0.925 → 0.950 → 0.942 |
| **M-next + clip + `slot_gain_lambda` 100 @ 0.9** | **forward, on the map** | **clean to 5000, 0 spikes** | **0.887–0.897 at every checkpoint** |

Cost: about 5 % wall clock (110 against 115 steps/min). Validation at 5000 (mean of the
last four evals): renorm 4.276, penalty 4.281, against the healthy memory-target arm's
4.308 and the coreless 8-layer floor's 4.079.

## 4. What it does NOT do

It does not make the loop earn. Both stable arms read forecast-loss K1−K6 0.013 and K3−K6
0.000 at 5000: the loop's work is done in three iterations, and the token loss reads
0.0006 of it. Stability and contribution are separate axes. Do not present this file's
result as a depth result.

## 5. What NOT to reach for (all measured, all refuted)

- **Ternary off.** It removes the drift (clean 3/3 on the paid axis, 1/1 here) and is not
  a design: ternary is the recipe (Wolfe, 2026-09-02, `no-dense-warmup-before-ternary`).
- **A weight-spectrum cap or projection** on the core weights: four variants failed on the
  takeover, and the capture shows why — every block's typical gain sits at 1.00–1.02 while
  the six-block map's worst gain goes 3 → 500. A bound on one factor cannot see the
  alignment of six.
- **The backward clip alone** (`slot_cot_clip` without a forward lever): the carried
  cotangent is bounded and the weight-path gradient grows 800x on an inflating forward.
- **`core_gain_clip`** (the L1 realised-magnitude governor): a symptom clamp, refuted on
  the takeover; it is not this constraint.
- **A dense warmup, γ-EMA, γ-freeze, GLA as a stabiliser**: section A of the divergence
  README.

## 6. Where everything is

- Code: `morph/model/transformer.py` — `MORPHConfig.slot_gain_*`, `_slot_gain_penalty`,
  `_loop_cot_hook` (the clip), the renorm inside `_tul_core`; `morph/training/train.py`
  reports the model loss without the penalty (the sigreg contract) and logs `tul/gain_est`.
- Tests: `tests/test_slot_gain_reg.py`, `tests/test_slot_cot_clip.py`,
  `tests/test_onset_capture.py`.
- Records, in order: `lab/experiments/successes/2026-09-03-tul-onset-capture.md` (the
  mechanism, bit-exact replay), `failures/2026-09-04-tul-clip-through-time.md` (the clip
  alone, and A1 at full BPTT), `successes/2026-09-04-tul-forward-levers.md` (the two
  levers), `failures/2026-09-03-tul-think-once-panel.md` (the panel that exposed it).
- Instruments: `lab/divergence/jac_sweep.py` (fp32 gain over checkpoints on one batch),
  `lab/divergence/cot_calibrate.py`, `lab/divergence/onset_locate.py`,
  `training.loop_cot_probe`, `training.loop_rank_every`, rolling checkpoints.
- The wider map of divergence faces: `lab/divergence/DIVERGENCE-README.md` (section D).
- Design: `.agents/notes/implemented/architecture/2026-09-04-slot-loop-gain-constraint.md`
  and the open contribution question in
  `.agents/notes/proposed/architecture/2026-09-04-loop-contractivity-as-design.md`.
