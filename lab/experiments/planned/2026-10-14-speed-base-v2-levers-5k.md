# Speed base v2 at 5k: diet stack + HC backward on tensor cores + glue folds

Status: planned

## Question

The FLOP diet stack (`nowta_pointer_fast_spdcells_speed_bf16c_cellp2_dec1`, one-block span decoder
+ cell mean 2) passed at 3 seeds (`successes/2026-10-13-flop-diet-5k.md`). Two rounding-only speed
keys merged the same night, each gated on the old parent:

- `model.hc_bwd_fused` (merge 9e0e8517): the HC entry backward's projection term on tensor cores.
  Bench on the parent: 158.50 -> 157.01 ms.
- `model.glue_fold` (merge e4b5d738): compiled glue (span decoder norms + SwiGLU, pointer mass,
  tied-head table, injection term, diagonal injection) and an exact pad + cast. Bench on the
  parent: 160.67 -> 151.53 ms.

Do both keys together on the stack keep CE@6 and K1-K6, and what is the new step time?

## Hypothesis

Both keys change rounding only (each bounded against fp64 in its tests), so the trained model is
the stack's within seed noise. The glue fold's saving shrinks a little on the stack, because the
one-block decoder has half the decoder glue.

## Predictions

Control: the stack's own 3 runs, same seeds (seed 1 3.8999 / +0.0051, seed 2 3.8939 / +0.0096,
seed 3 3.9058 / +0.0044; mean 3.8999 / +0.0064).

| Arm | CE@6 best guess (3-seed mean) | K1-K6 best guess | bench ms, pinned pass 3 |
|---|---|---|---|
| stack (bench only) | | | 130 |
| stack + both keys | 3.900 | +0.0062 | 121 |

Decision rule (Wolfe's paired reading for rounding-only changes, 2026-10-13):

- PASS if the mean seed-paired CE@6 difference is <= +0.003 AND the 95 % t-interval of the three
  seed-paired K1-K6 differences includes zero or lies above it.
- FAIL otherwise, or on a stability miss: `tul/gain_est` above 0.95 after step 60, or grad-norm
  max/median above 30 (one recovered spike inside the 1000-step warmup is reported, not failed).
- The cell knob (M1-M4, M2-M4) is reported, not gated.

If it passes, this config becomes the speed base and the morning report says so.

## Method

Master at the commit adding this file, `git archive` to `abq/tree-lev2`, per-tree
`TORCHINDUCTOR_CACHE_DIR` (a shared cache crashed another tree's gates on 2026-10-14).
`abq/chain-lev.sh`: (1) the key-off byte gate on this tree (45 deterministic graph steps of
`nowta_pointer_fast_spdcells`, FINAL_SHA256 must be 4a73ebf5...). (2) `bench_step.py`, arms stack /
stack + both keys, `model.slot_gain_pass_force=3`, 300 steps, skip 100, 2 repeats. (3) Seeds 1-3,
5000 steps, then `core_depth_sweep.py --depths 1,6 --rows 480 --batch 3`, `stab.keep.py`, and
`cell_count_sweep.py --cells 1,2,4 --cell-extend cyclic`.

## Results

## Verdict

## Updated hypothesis
