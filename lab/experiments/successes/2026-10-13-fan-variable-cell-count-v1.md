# Does a variable cell count train a cheaper fan with a working cell-count knob?

Status: success

Design note: `.agents/notes/proposed/architecture/2026-10-12-fan-variable-cell-count-design.md`.
Code: `tul.cell_count` (merge 33113fb6, default `fixed`).

## Question

The ungraded fan trains and runs 4 cells per slot. If each training batch draws its cell count
M (M <= 4), three things could change:

1. the training step gets cheaper (the loop, its glue and the hinge scale with M);
2. the model works at fewer cells at inference (a cheap decode mode);
3. the model works at MORE cells than it trained with (Wolfe: "can I train at 4 and inference
   at 128 fine?").

Do the draws keep the fixed-four fan's CE and loop contribution at M = 4, and does either arm
gain CE when M at inference goes below or above 4?

v1 cannot fully answer question 3. Each trained cell has a role (cell 0 is the copy cell; the
four `W_prefix` rows sit 0.38-0.49 apart), so a cell past the role table is a copy of a role or
an untrained one. Exchangeable cells (design note v2) are the real test of question 3.

## Hypothesis

A cell draw acts as dropout over cells. Training at fewer cells costs a little CE at M = 4,
because each role trains on fewer steps, and makes the model much less sensitive to M < 4. Above
4 no arm gains, because the cells have roles and the coda's softmax read and positions stretch
with M.

## Predictions

Arms:

- **fixed four cells** (control, not rerun): the filed bf16-carrier runs, seeds 1 and 2, CE@6
  3.9060 / 3.9040 (mean 3.9050), K1-K6 +0.0058 / +0.0052 (mean +0.00550)
  (`successes/2026-10-12-bf16-carrier-matched-controls.md`). They stand as controls because the
  key-fixed byte gate on the bf16c config equals master 41be5b52 (666ac56a), and S1 + S2 are exact.
- **gentle cell drop** (Wolfe's milder law): `nowta_pointer_fast_spdcells_speed_bf16c_cellg375`,
  P(M=3) 0.25, P(M=4) 0.75, a random subset of roles. Seeds 1 and 2.
- **Poisson cell drop**: `..._cellp3`, Poisson(3.5) clamped to [1, 4] (P = 0.136 / 0.185 /
  0.216 / 0.463 for M = 1..4), roles 0..M-1. Seeds 1 and 2.

A number is the mean of an arm's two seeds. CE@6 is CE at slot depth 6 on the 480 sweep rows.
gap(M) = CE@6(M) - CE@6(4) of the same checkpoints on the same rows (packed once, re-laid out per M).

**Cost.** Measured step FLOPs (builder G5, eager, depth 6): M = 1..4 gives 10.20 / 11.57 / 13.01 /
14.38 TFLOP.

| Reading | gentle cell drop | Poisson cell drop |
|---|---|---|
| expected step TFLOP | 14.04 (-2.4 %) | 13.00 (-9.6 %) |
| weighted wall vs control (pinned per-M bench) | 0.98x (band 0.96-1.00) | 0.92x (band 0.88-0.96) |

**Quality at M = 4** (difference to the control means):

| Reading | gentle cell drop | Poisson cell drop |
|---|---|---|
| CE@6 difference | +0.001 (band -0.003 to +0.005) | +0.004 (band -0.002 to +0.010) |
| K1-K6 / control | 98 % | 91 % |

**Downward (gap(M), M < 4):**

| M | fixed four cells | gentle cell drop | Poisson cell drop |
|---|---|---|---|
| 3 | +0.012 | +0.003 | +0.004 |
| 2 | +0.030 | +0.020 | +0.012 |
| 1 | +0.070 (band +0.03 to +0.12) | +0.055 | +0.035 (band +0.015 to +0.06) |

**Upward (gap(M), M > 4, rule `cyclic` unless named):** every arm +0.005 to +0.010 at M = 8,
+0.015 to +0.030 at M = 16; `fresh` at M = 8 worse than `cyclic` by about 0.01.
P(any arm's gap(8, cyclic) <= -0.002 with both seeds' paired 95 % CI below 0) = 15 %.

**Stability, every run:** grad-norm max/median <= 30; `tul/gain_est` inside 0.85-0.95 from step
1000 to 5000.

### Decision rules (2-seed means)

1. **Gentle cell drop goes into the speed base** if CE@6 difference at M = 4 <= +0.003, K1-K6 >=
   95 % of the control (>= +0.00523), and stability holds. Its FLOP saving is small (2.4 %), so the
   case for it is the cheaper M = 3 inference: the note recommends M = 3 at inference if its
   gap(3) <= +0.005.
2. **Poisson cell drop becomes the speed base** (deployed at M = 4) if CE@6 difference <= +0.003,
   K1-K6 >= 95 %, weighted wall <= 0.94x, and stability holds.
3. The K1-K6 line is tighter than the measurement (two seeds of one config spread 0.0023). If CE@6
   holds and K1-K6 falls between +0.0032 and +0.00523, the verdict is "not resolved at n=2" and
   the next step is a third seed of that arm and the control. Below +0.0032 the arm fails on K1-K6.
4. **Downward knob live** for an arm if gap(1) <= +0.040 and is at least 0.020 below the control's.
5. **Upward knob** for an arm if gap(8, cyclic) <= -0.002 and both seeds' paired 95 % CIs for
   CE(8) - CE(4) lie below 0. If no arm passes, v1 does not extend past training; the next step is
   v2 (exchangeable cells, pooled coda read) or training at M up to 8 (needs more memory). Wolfe picks.
6. Cosines are reported and decide nothing.

## Method

1. Master at the commit adding this file, copied with `git archive` to
   `/home/wolfe/morph-scratch/abq/tree-cc`. First a 45-step deterministic graph-step byte gate of
   `nowta_pointer_fast_spdcells` on that tree: FINAL_SHA256 must be `4a73ebf5...bd70d7` (merge of
   S5 and the cell count, both keys off). If it does not match, stop.
2. `abq/chain-cc.sh`: gentle s1, gentle s2, Poisson s1, Poisson s2, 5000 steps each with
   `training.seed=<s>`, one at a time under `gpu.lock`. After each: `core_depth_sweep.py --depths
   1,6 --rows 480 --batch 3` and `abq/r2bench/stab.keep.py`.
3. Cell sweep with `lab/divergence/cell_count_sweep.py --rows 480 --batch 3` on the four new
   checkpoints and the two controls (tree-n `ab-bf16b`, `ab-bf16-s2`): `--cells 1,2,3,4,8,16
   --cell-extend cyclic` at `--depth 6` and `--depth 1`, plus `--cells 4,8 --cell-extend fresh
   --depth 6`. The M = 4 point must equal each run's `core_depth_sweep` CE@6 to 1e-4, or the sweep
   is wrong and no gap is read. M = 32 is left out (the coda row at 32 cells is 1024 + 64 x 32 =
   3072 positions, far past the 1280 it trained on, and it costs most of the sweep time).
4. Cost: `lab/perf/bench_step.py`, every arm with `model.slot_gain_pass_force=3`, 300 steps,
   skip 100, 2 interleaved repeats. Arms: the control (the fixed bf16c config, M = 4) and M = 1,
   2, 3 forced on it with `+tul.cell_count=categorical` and one-hot `+tul.cell_count_probs`.
   Weighted wall = sum over M of P(M) x ms(M) / ms(control).
5. Fill Results, Verdict and Updated hypothesis; move the file to `successes/` or `failures/`.

## Results

Log `abq/chain-cc.log`, tree `abq/tree-cc` (master 6b833a5b). Byte gate on that tree (S5 and the
cell count both off): `4a73ebf5`, match. Every M = 4 sweep point equals that run's
`core_depth_sweep` CE@6 at the printed 4 decimals, so the cell sweep is read.

**Quality at M = 4** (480 rows):

| Arm | Seed | CE@1 | CE@6 | K1-K6 [95 % CI] | grad-norm max/median | gain_est 1000-5000 |
|---|---|---|---|---|---|---|
| fixed four cells (filed) | 1 | 3.9118 | 3.9060 | +0.0058 [+0.0053, +0.0062] | 6.6 | 0.889-0.902 |
| fixed four cells (filed) | 2 | 3.9093 | 3.9040 | +0.0052 [+0.0048, +0.0057] | 7.8 | 0.890-0.902 |
| gentle cell drop | 1 | 3.8951 | 3.8899 | +0.0051 [+0.0047, +0.0056] | 8.9 | 0.885-0.903 |
| gentle cell drop | 2 | 3.9222 | 3.9166 | +0.0056 [+0.0052, +0.0060] | 6.6 | 0.885-0.903 |
| Poisson cell drop | 1 | 3.9166 | 3.9117 | +0.0049 [+0.0044, +0.0053] | 7.4 | 0.881-0.904 |
| Poisson cell drop | 2 | 3.9043 | 3.8987 | +0.0056 [+0.0051, +0.0060] | 7.8 | 0.875-0.905 |

- Means: control 3.9050 / +0.00550; gentle 3.9033 / +0.00535 (CE@6 -0.0018, K1-K6 97 %);
  Poisson 3.9052 / +0.00525 (CE@6 +0.0002, K1-K6 95.5 %).
- The gentle seeds spread 0.027 on CE@6 (3.8899 and 3.9166), 8x the control seeds' spread
  (0.0032). Its mean passes, but this arm's draw spread is wide.

**Cost** (pinned bench on the bf16c base without S5, 2 repeats; ms/step and ratio to the control):
control 174.94; M = 1 132.57 (0.758); M = 2 146.85 (0.839); M = 3 161.82 (0.925).
Weighted wall: gentle 0.981x, Poisson 0.921x. Expected TFLOP (builder G5 per-M FLOPs): Poisson
13.00 (-9.6 %), gentle 14.04 (-2.4 %). Trainer tok/s: control 31.6k, gentle about 37k, Poisson 38-40k
(the control ran without S1 + S2; sanity check only).

**gap(M) = CE@6(M) - CE@6(4)**, mean of the two seeds (per-seed paired 95 % CIs in
`abq/cc-cells-d6.json`):

| M | fixed four cells | gentle cell drop | Poisson cell drop |
|---|---|---|---|
| 1 | +0.1346 | +0.0899 | +0.0051 |
| 2 | +0.0708 | +0.0252 | +0.0002 |
| 3 | +0.0252 | +0.0004 | -0.0001 |
| 8, cyclic | +0.0657 | +0.0635 | +0.0589 (+0.0137 / +0.1041) |
| 16, cyclic | +0.1191 | +0.1048 | +0.1363 |
| 8, fresh | +0.0530 | +0.0585 | +0.0600 (+0.0110 / +0.1089) |

- Poisson at M = 1: CE@6 3.9103 (control at M = 4: 3.9050, +0.0053); K1-K6 at M = 1 +0.0037 (67 %
  of the control's). At M = 2: CE@6 3.9054 (+0.0004), K1-K6 +0.00515 (94 %).
- Gentle at M = 3: CE@6 3.9036 (control at M = 4: 3.9050), K1-K6 at M = 3 +0.0054.
- No arm gains at any M > 4. Every M > 4 CI lies above 0.

## Verdict

Success. The predictions held on every decision rule; the downward curve came out far flatter
than predicted for the Poisson arm (gap(1) +0.005 against a guess of +0.035).

1. Gentle cell drop passes rule 1 (CE@6 -0.0018, K1-K6 97 %, stable) and its gap(3) is +0.0004:
   M = 3 at inference is free on it.
2. Poisson cell drop passes rule 2: CE@6 +0.0002, K1-K6 95.5 % (0.00002 over the line, so n = 2
   cannot separate it from a 5 % loss), weighted wall 0.921x, stable. It becomes the loop recipe's
   speed base, deployed at M = 4.
3. The downward knob is live on the Poisson arm (rule 4): gap(1) +0.0051, 0.130 below the control.
   M = 2 at inference is within 0.0004 CE of M = 4.
4. The upward knob fails on every arm (rule 5): more cells than trained cost +0.05 to +0.14 at
   M = 8-16, cyclic or fresh. v1 does not extend past training.

## Updated hypothesis

Trained with a random cell count, the fan learns cells that each carry nearly the whole read:
the coda loses 0.005 nats going from 4 cells to 1. So at 5k the extra cells are worth about 0.005
nats at eval, and the fixed-four fan's 0.135 at M = 1 is co-dependence of its cells, not value.
Whether the extra cells' value grows with training (Wolfe's crossover hypothesis, about 2B
tokens) is open. "Train at 4, infer at 128" needs exchangeable cells and a pooled coda read
(design note v2); v1's role-indexed cells and the coda's position range do not extend.
