# S5 block custom ops at 5k against matched bf16 controls

Status: planned

## Question

`model.block_custom_ops` (S5, master 33e52957) makes each block compile as one graph: 1.0229 on
the speed base (171.70 -> 167.85 ms/step, 2 interleaved repeats, 2026-10-13). It is not exact:
each merged graph draws its own philox seed, so the dropout masks change (same law, different
draw). Does it train the same function at 5k, so it can go ON in the speed base?

## Hypothesis

S5 changes which units drop and nothing else (S5-only and S5 + S1 + S2 byte gates both
`3381bc7e`; fixed-weight readout compiled off and on identical, CE@6 3.8503829). A new mask draw
is a seed change, so the S5 means sit inside the seed spread of the controls.

## Predictions

Two new runs, `nowta_pointer_fast_spdcells_speed_bf16c` (S1 + S2 on, graphs kept, probe block
every 4th eval) + `model.block_custom_ops=true`, seeds 1 and 2. Controls are the existing bf16
carrier runs under the same eval regime (S1 and S2 are exact, the 45-step gate with both on
matches `4a73ebf5`): seed 1 CE@6 3.9060 / K1-K6 +0.0058, seed 2 3.9040 / +0.0052
(`successes/2026-10-12-bf16-carrier-matched-controls.md`). Control means: CE@6 3.9050, K1-K6
+0.00550.

| Arm | Seed | CE@6 best guess | K1-K6 best guess |
|---|---|---|---|
| S5 custom ops | 1 | 3.905 | +0.0055 |
| S5 custom ops | 2 | 3.905 | +0.0055 |

- Decision: S5 goes ON in the speed base if the S5 mean CE@6 minus 3.9050 is <= 0.003 AND the
  S5 mean K1-K6 is >= 95 % of +0.00550 (>= +0.00523).
- The K1-K6 line is tighter than the measurement. Between two seeds of one config K1-K6 spreads
  0.0023, so with 2 seeds per arm a neutral change misses the 95 % line about 4 times in 10.
  Pre-committed handling: if CE@6 holds and K1-K6 falls below +0.00523 but stays >= +0.0032
  (the control mean minus one seed spread), the verdict is "not resolved at n=2" and the next
  step is a third seed for both arms (control seed 3 on the bf16c config without S5). Below
  +0.0032, S5 stays off and the cause gets looked for.
- Stability on both runs: grad-norm max/median <= 30, `tul/gain_est` inside 0.85-0.95 from step
  1000 to 5000.
- Speed: trainer tok/s about 2 % above the bf16c seed-2 control's 31.6k (bench_step is the real
  number; the trainer rate is a sanity check only).

## Method

Master at the commit adding this file, copied with `git archive` to
`/home/wolfe/morph-scratch/abq/tree-s5`. `chain-s5.sh` in `abq/`: for seeds 1 and 2, 5000
steps with `training.seed=<s> model.block_custom_ops=true`, one at a time under `gpu.lock`, then
`lab/divergence/core_depth_sweep.py --depths 1,6 --rows 480 --batch 3` on the final checkpoint and
`abq/r2bench/stab.keep.py` on the wandb file (same as `chain-bf16ctl.sh`).
