# Triton GEMMs (G1) with S5 on the Poisson cell-count base at 5k

Status: planned

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
