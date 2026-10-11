# One-cell loop against LXTUL + pointer at 20k

Status: planned

## Question

At 5k steps on the fast recipe the one-cell loop beats LXTUL + pointer: CE@6 3.8997 against
3.9187-3.9316 (four LXTUL references), loop write zeroed 0.193 against 0.154 / 0.168, and only K1-K6
favours LXTUL (+0.0022 against +0.0088-0.0104). Wolfe's suspicion (2026-10-10): the one cell wins
early because its objective is easier, and the 4 cells win after more training (he named 2B tokens).
Does the gap narrow with training?

## Hypothesis

Two accounts.

- Crossover (Wolfe): the 4-cell selection learns slower and pays off later; the trainer-val gap
  (one-cell minus LXTUL, same val batches per step) shrinks toward 0 and turns positive.
- No crossover (mine, weakly held): the gap has been flat at -0.018 to -0.024 from step 2000 to
  5000 in the 5k pair, and on the older eager recipe without the pointer the one-cell strict loop
  (fp01) led LXTUL by 0.007 at 5k and 0.029 at 10k against the same plain ruler. The gap stays near
  -0.02 or widens.

20k steps is about 154M tokens, under a tenth of the 2B Wolfe named. This run can show the trend;
it cannot rule out a crossover later.

## Predictions

- Trainer-val gap (one-cell minus LXTUL) averaged over the last 4 evals: best guess -0.025.
  At or above -0.010 counts as narrowing (supports a crossover); at or below -0.030 counts as
  widening.
- The gap's slope over steps 10k to 20k: best guess flat (within +/-0.005 per 10k steps).
- CE@6 (480 rows) one-cell minus LXTUL: best guess -0.025.
- K1-K6 LXTUL above one-cell at 20k as at 5k: best guess LXTUL +0.012, one-cell +0.003.

Decision: narrowing at or above -0.010 by 20k, with a positive slope, is reason to run the pair
longer (50k or more). A flat or widening gap is reason to make the one-cell loop the base and to
test whether 4 DISTINCT cells (the each-cell decoder arm) change it.

## Method

`tul_onecell_pointer_fast` and `lxtul_pointer_fast`, both with `training.steps=20000`, from the
frozen tree of the commit that adds this file, seed 1, WANDB offline, run one after the other under
`gpu.lock` by `/home/wolfe/morph-scratch/abq/runner.sh`. The AdEMAMix beta3 horizon falls back to
`training.steps`, so both arms carry the same 20k horizon. Readout: trainer val at every eval step
(same val batches per step in both runs), and `core_depth_sweep.py --depths 1,6 --rows 480 --batch
3` on each final checkpoint.

Amended 2026-10-10: not queued. Wolfe asked first for an extrapolation of the 5k curves
(`lab/experiments/results/2026-10-10-lxtul-credit-panel/extrapolate_val_gap.py`). The run waits
for his word.
