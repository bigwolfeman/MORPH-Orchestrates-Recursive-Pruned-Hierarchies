# Triton GEMMs (G1) with S5 on the Poisson cell-count base at 5k

Status: success

## Question

`model.split_wgrad_triton` (G1, merge b4810c99) replaces the cuBLAS GEMMs inside `split_linear`
with one tuned Triton kernel: pinned bench on top of S5 171.07 -> 164.21 ms/step (-6.9 ms). It is
rounding only (another K order than cuBLAS), and the fixed-weight readout did not move (CE@6
3.9059 vs 3.9060). The speed base is now the Poisson cell-count fan (`..._speed_bf16c_cellp3`),
which inherits S5 since 2026-10-13, but its filed 5k runs ran without S5. Does the base with S5
and G1 both on train the same function at 5k?

## Hypothesis

G1 changes rounding only, and S5 only redraws dropout masks (passed alone:
`successes/2026-10-13-s5-custom-ops-5k.md`). Together they sit inside the seed spread.

## Predictions

New arm: `nowta_pointer_fast_spdcells_speed_bf16c_cellp3` (S5 on) +
`model.split_wgrad_triton=true`, seeds 1 and 2. Controls: the filed Poisson cell-drop runs
(no S5, no G1), seed 1 CE@6 3.9117 / K1-K6 +0.0049, seed 2 3.8987 / +0.0056 (means 3.9052 /
+0.00525; `successes/2026-10-13-fan-variable-cell-count-v1.md`).

| Arm | Seed | CE@6 best guess | K1-K6 best guess |
|---|---|---|---|
| S5 + G1 on the Poisson base | 1 | 3.905 | +0.0053 |
| S5 + G1 on the Poisson base | 2 | 3.905 | +0.0053 |

- Decision: G1 goes ON in `nowta_pointer_fast_spdcells_speed_bf16c` (so the Poisson base inherits
  it) if the arm's mean CE@6 minus 3.9052 is <= +0.003 AND its mean K1-K6 is >= 95 % of +0.00525
  (>= +0.00499). If CE@6 holds and K1-K6 sits in [+0.0030, +0.00499): "not resolved at n=2", a
  third seed of the arm and of the control. Below +0.0030: G1 stays off and the cause is looked for
  (S5 alone already passed, so G1 is the suspect).
- The cell knob survives: gap(1) mean <= +0.015 and gap(2) mean <= +0.003 (controls +0.0051 /
  +0.0002).
- Stability on both runs: grad-norm max/median <= 30, `tul/gain_est` inside 0.85-0.95 from step
  1000 to 5000.
- No run raises at a graph recording (the G1 re-record path); if one re-records, its log line
  is reported.

## Method

Master at the commit adding this file, `git archive` to `/home/wolfe/morph-scratch/abq/tree-g1`.
`abq/chain-g1.sh`: (1) the GPU test files `tests/test_stream_overlap.py` and
`tests/test_graph_step.py` on that tree under `gpu.lock`; stop on any failure. (2) The 45-step
key-off byte gate (`4a73ebf5...`); stop on mismatch. (3) Seeds 1 and 2, 5000 steps each, then
`core_depth_sweep.py --depths 1,6 --rows 480 --batch 3` and `stab.keep.py`. (4)
`cell_count_sweep.py --cells 1,2,4 --depth 6 --rows 480 --batch 3` on both new checkpoints.

## Results

Log `abq/chain-g1.log`, tree `abq/tree-g1` (master d19cb5d5). On that tree, before the runs:
`test_stream_overlap.py` 37 passed and `test_graph_step.py` 24 passed on the GPU; key-off byte
gate `4a73ebf5` (match).

| Arm | Seed | CE@1 | CE@6 | K1-K6 [95 % CI] | grad-norm max/median | gain_est 1000-5000 |
|---|---|---|---|---|---|---|
| Poisson base (filed, no S5, no G1) | 1 | 3.9166 | 3.9117 | +0.0049 [+0.0044, +0.0053] | 7.4 | 0.881-0.904 |
| Poisson base (filed, no S5, no G1) | 2 | 3.9043 | 3.8987 | +0.0056 [+0.0051, +0.0060] | 7.8 | 0.875-0.905 |
| S5 + G1 on the Poisson base | 1 | 3.9080 | 3.9023 | +0.0057 [+0.0053, +0.0062] | 7.6 | 0.878-0.905 |
| S5 + G1 on the Poisson base | 2 | 3.9100 | 3.9033 | +0.0067 [+0.0062, +0.0072] | 7.3 | 0.883-0.904 |

- Means: arm CE@6 3.9028, K1-K6 +0.0062; control 3.9052, +0.00525.
- Arm - control CE@6: -0.0024 (rule <= +0.003: held). K1-K6 118 % (rule >= 95 %: held).
- Cell knob: gap(1) +0.0059 / +0.0059 (mean +0.0059, rule <= +0.015: held); gap(2) -0.0001 /
  +0.0002 (mean +0.0001, rule <= +0.003: held).
- Stability held on both runs. Neither run re-recorded a graph (the G1 untuned-key path was not
  reached; every key was tuned in the warm-up).
- Trainer tok/s 40.5-42.3k (sanity check; the controls read 38-40k without S5 and G1).

## Verdict

Success: the predictions held. G1 goes ON in `nowta_pointer_fast_spdcells_speed_bf16c`, so the
Poisson speed base inherits S5 and G1.

## Updated hypothesis

G1 is rounding only at 5k: S5 + G1 on the Poisson base sit inside the seed spread on CE@6 and
K1-K6, and the cell-count knob is unchanged.
