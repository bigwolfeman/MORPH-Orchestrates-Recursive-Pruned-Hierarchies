# FAST2 package plus online teacher, paired against production

Status: mixed

## Question

Does the whole speed package as committed on perf/graph-step at efb5bf7a (graph-captured step,
compiled blocks, fused TG attention, HC fused grad, ternary step cache, no checkpoint recompute,
compiled fan twin, fp32 fan diversity terms, bf16 lsel pick) plus the online-prelude router
teacher keep the loop's contribution within the owner's 5 % rule over a 1000-step continuation?

## Hypothesis

Every key but the online teacher changes rounding only; rounding chaos does not move a 1000-step
continuation outside the paired noise. The online teacher changes the router's target by at most
the EMA lag (teacher pick agrees 93-98 % per pass), and the model CE does not depend on it while
`fan_lsel_train_follow` is `router`.

## Predictions

Baseline: the two no-change continuations of the same checkpoint (`mixed/2026-10-08-paired-continuation-noise.md`):
K1-K6 +0.0086 / +0.0091, CE@6 4.0461 / 4.0468.
- K1-K6 (ce_tokens, 480 rows) of each package continuation within 0.001 of the baseline mean
  (+0.00885): best guess +0.0085.
- CE@6 within 0.003 of the baseline mean (4.04645).
- Pass rule: both package continuations inside both bounds. One outside: rerun that arm once more.
  Both outside in the same direction: the package fails; bisect online teacher vs the rest next.

## Method

Two continuations, c and d, from `checkpoints/morph/lxtul-pointer-rerun/step_2500.pt`,
`training.resume=<ckpt> training.steps=3500`, recipe FAST2 (`/home/wolfe/morph-scratch/perf/graph/BRIEF.md`)
plus `model.fan_target_online=true`, tree = git archive of efb5bf7a, WANDB offline. Readout as the
baseline: `core_depth_sweep.py --depths 1,6 --rows 480 --batch 3` on each final checkpoint.

Method amended 2026-10-08 05:33: the tree is 31542ea9, not efb5bf7a. The first launch failed at
load (an eager checkpoint could not resume into a `compile_blocks` model); 31542ea9 fixes the key
alignment and changes nothing else on the training path. A second launch was stopped at step 2600
by mistake (its logged total loss 10.17 was read as CE; the eager continuations log 10.16 there).


## Results

Artifacts: [`results/2026-10-08-fast2-package-paired/`](../results/2026-10-08-fast2-package-paired/).
Every run resumed "model+scaler+RNG from checkpoint step 2500". A third eager continuation (f)
was added to the baseline after d came out high.

| run | CE@1 | CE@6 | K1-K6 (ce_tokens, 480 rows) | trainer val |
|---|---|---|---|---|
| eager a | 4.0547 | 4.0461 | +0.0086 | 4.0901 |
| eager b | 4.0559 | 4.0468 | +0.0091 | 4.0906 |
| eager f | | 4.0462 | +0.0098 | 4.0904 |
| package c | 4.0549 | 4.0461 | +0.0088 | 4.0901 |
| package d | 4.0656 | 4.0457 | +0.0199 | 4.0897 |
| package e (rerun of the arm) | 4.0553 | 4.0471 | +0.0082 | 4.0919 |

- CE@6: all three package runs within 0.001 of the baseline mean 4.04645. Held.
- K1-K6 within 0.001 of +0.00885: c (+0.0003) and e (-0.0007) held; d (+0.0111) missed, HIGH.
  By the pass rule written before the run, d was rerun once (e) and e is inside.
- Speed of the package arm in these runs: about 19.2k tok/s (trainer log), against 11.5k for the
  eager production arm on the same rows.

## Verdict

Mixed: the package passes the preregistered rule (CE unchanged in all three runs; two of three
K1-K6 draws inside the band) and one draw broke the K1-K6 prediction on the high side. Run d did
not lose loop contribution: its depth-6 CE is the same as every other run's and its depth-1 CE
is 0.011 worse, so it learned a more depth-dependent solution.

## Updated hypothesis

The package does not damage CE or reduce loop contribution over a 1000-step continuation. Its
K1-K6 spread over three draws (+0.0082 to +0.0199) is wider than the eager spread (+0.0086 to
+0.0098); three draws cannot say whether the package widens the spread or d is a rare draw.
Side finding (not a package difference): `_tul_core` draws the gain hinge's pass from the CPU
generator and restores it, and `graph_step` raises if the CPU generator moves between replays.
No run raised, so in eager production too the hinge reads ONE pass for a whole run, chosen by
the seed, not a fresh pass per step. Every winner number trained that way.
