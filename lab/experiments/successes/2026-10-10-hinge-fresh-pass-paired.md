# Fresh gain-hinge pass per step, paired against the fixed pass

Status: success

## Question

The slot loop's gain hinge (`model.slot_gain_lambda` 100 at target 0.9) reads ONE loop pass per
step. Until 2026-10-10 that pass was the same for a whole run (the draw put the global CPU stream
back: 54 of 54 draws were pass 6 on seed 1, pass 3 on seed 2; phase-2 tracker A1). Wolfe decided a
fresh pass per step (tracker A2), now `model.slot_gain_pass_force: -1`. Over a 1000-step
continuation of the production checkpoint, does the fresh draw keep CE@6 and the loop's
contribution (K1-K6) where the fixed pass had them?

## Hypothesis

The hinge is a barrier: at this checkpoint it binds on about a third of the logged steps (fixed
arm: `tul/gain_reg_weighted` > 0 on 0.347 / 0.347 / 0.306 of 49 log steps, `tul/gain_est` mean
0.868 / 0.866 / 0.868) and the slot map's measured gain sits just above the injection floor (0.865; root
CLAUDE.md, 2026-09-24; not measured per pass), so moving the read across the 8 passes changes
which pass the barrier guards more than the map the loss trains. CE@6 and K1-K6 stay inside the
fixed arm's own three-draw spread.

## Predictions

Baseline: the three fixed-pass continuations filed in
`lab/experiments/successes/2026-10-08-slot-compaction-paired.md` (same checkpoint, same recipe,
same readout): CE@6 4.0480 / 4.0468 / 4.0472 (mean 4.04733); K1-K6 (ce_tokens, 480 rows) +0.0098 /
+0.0093 / +0.0120 (mean +0.01037); trainer val 4.0926 / 4.0915 / 4.0920. Their hinge read pass 6
on every step: the pass `torch.randint(8)` draws under the checkpoint's saved CPU state (computed
2026-10-10 from `rng_cpu` of step_2500.pt; those runs did not log it). The eager paired noise
over three draws is 0.0012 in K1-K6 (`mixed/2026-10-08-paired-continuation-noise.md`); the fixed
arm's own spread is 0.0027.

- CE@6 of each fresh continuation within 0.002 of 4.04733; best guess 4.0474.
- Mean K1-K6 of the three fresh continuations at or above +0.0077 (no drop larger than the fixed
  arm's own spread below its mean, 0.0027) and at or below +0.0131; best guess +0.0100.
- The draw moves: each run logs `loop/gain_pass` on all 1000 steps, every pass 0-7 appears, and
  the histogram passes a chi-square uniformity check (chi2 < 24.32, 7 dof, p = 0.001). The three
  runs log the same pass sequence (same seed, checkpoint without a draw state).
- Secondary reading, not a pass rule: the share of log steps with `tul/gain_reg_weighted` > 0
  between 0.15 and 0.60; best guess 0.35.

Resolution, stated now: three pairs cannot resolve the owner's 5 % of K1-K6 (about 0.0005); this
test detects a K1-K6 shift of about 0.0015-0.0027 or more and a CE@6 shift of about 0.001-0.002.
A pass means "no shift that large", not "within 5 %".

Decision rule: both CE@6 and K1-K6 predictions hold -> the fresh draw stays the default
(`slot_gain_pass_force: -1`, already the default in the tree) and `lxtul_pointer_fast` trains with
it. CE@6 misses on one run only -> rerun that run once. Mean K1-K6 below +0.0077 or a CE@6 miss on
two runs -> the fresh draw costs the loop; report to Wolfe with the per-pass hinge readings
(`loop/gain_pass` against `tul/gain_est` at the log steps) before any change.

## Method

Tree: `git archive` of the commit that adds this file (it carries the draw:
`MORPHTransformer._gain_pass_for_step`, `draw_gain_pass`, GraphStep variants). Three
continuations, `fresh-1`, `fresh-2`, `fresh-3`, from
`/home/wolfe/morph-wt-graph/checkpoints/morph/lxtul-pointer-rerun/step_2500.pt`,
`--config-name lxtul_pointer training.resume=<ckpt> training.steps=3500`, the paired3 recipe key
for key (FAST2 + `model.fan_target_online=true` + `model.slot_depth_stratified=true
model.slot_compact=gather`) plus `model.slot_gain_pass_force=-1` (the default, passed
explicitly), WANDB offline, under `gpu.lock`. The checkpoint predates the draw, so each run's
draw starts from training.seed 1 (the trainer prints `[gain-pass] the checkpoint predates ...`).
Under `training.graph_step` each step draws its pass on the host and replays the graph recorded
for it (16 graphs: 2 step kinds x 8 passes, recorded again after every eval).

Readout: `lab/divergence/core_depth_sweep.py --depths 1,6 --rows 480 --batch 3` on each final
checkpoint, the baseline's command. Pass histogram: `read_passes.py` on each run's offline
wandb (the chain does it). Chain script, ready and not launched:
`/home/wolfe/morph-scratch/hingedraw/paired/chain.sh <commit>`. Artifacts to
`lab/experiments/results/2026-10-10-hinge-fresh-pass-paired/` when filed.

## Results

Run 2026-10-10 11:15-11:49, tree `git archive be624816` (e19234c2 + the warm-up slot_id remap;
the draw is unchanged). Artifacts: `lab/experiments/results/2026-10-10-hinge-fresh-pass-paired/`
(`chain.txt`: trainer, pass-histogram and readout lines; `passes_*.txt`: every logged pass).

| Run | CE@6 | K1-K6 (ce_tokens, 480 rows) | trainer val | hinge active share | gain_est mean |
|---|---|---|---|---|---|
| fresh 1 | 4.0482 | +0.0117 [+0.0110, +0.0124] | 4.0942 | 0.347 | 0.8764 |
| fresh 2 | 4.0480 | +0.0121 [+0.0114, +0.0128] | 4.0938 | 0.327 | 0.8769 |
| fresh 3 | 4.0487 | +0.0124 [+0.0117, +0.0132] | 4.0941 | 0.388 | 0.8770 |
| fresh mean | 4.04830 | +0.01207 | 4.0940 | 0.354 | 0.8768 |
| fixed pass 6 (baseline mean) | 4.04733 | +0.01037 | 4.0920 | 0.333 | 0.867 |

Pass draw: 999 logged steps per run (2501..3499), counts over passes 0..7 = 108, 119, 135, 129,
131, 119, 130, 128, chi2 4.38 (7 dof), the same sequence in all three runs. 16 graphs recorded
4 times per run, 15-16 s in total.

Against the predictions:
- CE@6 within 0.002 of 4.04733: +0.0009, +0.0007, +0.0014. Held (best guess 4.0474; the
  fresh runs sit 0.001 above the baseline on all three, the same sign as the trainer val's
  +0.002).
- Mean K1-K6 in [+0.0077, +0.0131]: +0.01207. Held (best guess +0.0100). The shift from the
  baseline mean is +0.0017, inside the fixed arm's own spread (0.0027) and just above the
  paired noise (0.0012).
- The draw moves: every pass 0-7 appears, chi2 4.38 < 24.32, the same sequence in the three
  runs. Held, with one literal miss: the prediction said all 1000 steps log `loop/gain_pass`;
  999 do (step 3500 does not log it; the cause is not checked, the draw itself is per step).
- Secondary, share of log steps with `tul/gain_reg_weighted` > 0 in [0.15, 0.60]: 0.327-0.388.
  Held (best guess 0.35).

## Verdict

Success. A fresh pass per step keeps CE@6 and the loop's contribution where the fixed pass had
them, at this test's resolution (CE@6 about 0.001-0.002, K1-K6 about 0.0015-0.0027). The fresh
draw stays the default (`model.slot_gain_pass_force: -1`) and `lxtul_pointer_fast` trains with
it once perf/hinge-draw merges. Cost: +2.0 % per training step (two 420-step benches per tree).

## Updated hypothesis

The barrier picture holds: the hinge is active on about a third of the log steps whichever pass
it reads. Reading all passes moves the mean `gain_est` from 0.867 to 0.877, because passes other
than 6 carry a slightly higher gain; the map the loss trains does not move enough to change CE@6
or K1-K6 at 1000 steps. Small consistent signs to watch at longer horizons: CE@6 +0.001 and
trainer val +0.002 (all three runs), K1-K6 +0.0017. `model.core_gain_lambda` (`_core_region`)
still reads a frozen pass; no decision covers it yet.
