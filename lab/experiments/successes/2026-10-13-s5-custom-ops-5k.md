# S5 block custom ops at 5k against matched bf16 controls

Status: success

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

## Results

Log `abq/chain-s5.log`, tree `abq/tree-s5` (master 08b2d966). Sweep: 480 rows, batch 3.

| Arm | Seed | CE@1 | CE@6 | K1-K6 [95 % CI] | grad-norm max/median | gain_est 1000-5000 |
|---|---|---|---|---|---|---|
| bf16 control (filed) | 1 | 3.9118 | 3.9060 | +0.0058 [+0.0053, +0.0062] | 6.6 | 0.889-0.902 |
| bf16 control (filed) | 2 | 3.9093 | 3.9040 | +0.0052 [+0.0048, +0.0057] | 7.8 | 0.890-0.902 |
| S5 custom ops | 1 | 3.9024 | 3.8973 | +0.0051 [+0.0046, +0.0055] | 7.8 | 0.886-0.901 |
| S5 custom ops | 2 | 3.9094 | 3.9033 | +0.0061 [+0.0056, +0.0066] | 7.4 | 0.890-0.902 |

- Means: S5 CE@6 3.9003, K1-K6 +0.0056; control 3.9050, +0.00550.
- S5 - control CE@6: -0.0047 (rule <= +0.003: held). K1-K6 ratio 102 % (rule >= 95 %: held).
- Stability held on both runs.
- Trainer tok/s 36.8k / 36.7k (controls 31.6k, but they ran before S1 + S2 were on; the trainer
  rate is a sanity check, the bench is the speed number: 1.0229).
- The seed-1 S5 run reads 0.0087 below its control on CE@6. A mask redraw is a seed change, so this
  is draw spread, not a gain from S5.

## Verdict

Success: the predictions held. S5 goes ON in `nowta_pointer_fast_spdcells_speed_bf16c`.

## Updated hypothesis

S5 trains the same function as the base at 5k (a mask redraw, inside the seed spread) and saves
2.2 % of the step.
