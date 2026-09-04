# Planned: clip-through-time on the forecast arm, and the A1 pair at full BPTT

Status: planned
Date: 2026-09-04 (frozen before any arm ran; the 12-step smokes and the per-row
calibration in Method precede launch and produce no metrics). Branch `tul/think-once`,
worktree `/home/wolfe/morph-to`, code `4d5a986`. Trigger: Wolfe, 2026-09-04 — "we can
run path 1 then 2. take all the runs you want to do, put them in the task list. then do
them." Path 1 = `.agents/notes/proposed/architecture/2026-09-04-loop-contractivity-as-
design.md` candidate (c): bound the backward product, forward untouched.

## Question

The onset capture (`successes/2026-09-03-tul-onset-capture.md`) measured the forecast
spike train as the cotangent growing 39–2436x back through the eight slot-loop iterations
on spike steps (1.6–3x calm) while the forward stays flat, with the loop's typical gain
drifting from 0.87 to 1.00 under ternary QAT and the forecast target. Every healthy
checkpoint in the sweep carried a product of 1.4–2.1 at any gain. Three questions:

1. Does bounding the product per row at 4x the exit cotangent (`model.slot_cot_clip:
   4.0`, `morph/model/transformer.py::_loop_cot_hook`) let the M-next arm reach 5000
   steps under the tripwire, on the recipe that tripped it at 3618 (fused) and 1208
   (eager)?
2. If it survives, does the bounded loop still do work: is the gain drift still there,
   and does the forecast loss earn depth (the panel's "thinks" rule)?
3. Is the truncation at `bptt_depth` 4 what keeps A1's product flat (1.4–1.7 at a typical
   gain of 0.975 and a worst gain of 176)? A1 at `bptt_depth` 8, two seeds, is the direct
   test, and it is Wolfe's contribution question from 2026-09-03 ("if the loop contributes
   now even if there are stability issues").

## Arms (configs `morph/configs/`; every arm composes `tul_to_panel.yaml`)

| # | run | config | one line |
|---|---|---|---|
| X1 | `to-mnext-ctt` | `tul_to_mnext_ctt` | R4 (M-next) + `slot_cot_clip` 4.0, cotangent probe on, checkpoints every 500, tripwire |
| X2 | `to-a1-b8-s1` | `tul_to_a1_b8` | A1 at `bptt_depth` 8, seed 1, checkpoints every 500, tripwire |
| X3 | `to-a1-b8-s2` | `tul_to_a1_b8` + `training.seed=2` | the second seed |

Controls already on disk: `to-mnext` (R4 unclipped, fused, tripped 3618), `cap-c1-det`
(R4 unclipped, eager, tripped 1208), `to-a1-s1`/`to-a1-s2` (A1 at bptt 4: one clean, one
takeover at 2041).

## Readouts

- The tripwire (`preclip/total > 1e4` at step ≥ 200) on every draw.
- From `probe.jsonl` (every step): `loop/cot_ratio` (pre-clip product, first grad
  iteration over the exit), `loop/cot_post_ratio` (what the backward carried),
  `loop/cot_bind_t{k}` and `loop/cot_bind_max` (fraction of rows the cap shrank).
- `lab/divergence/jac_sweep.py` over each arm's checkpoints (every 500 steps) on the
  capture's fixed batch: `jac/rms_t3`, `jac/sigma_t3`, the trajectory step ratios.
- `core_depth_sweep.py` (480 rows, batch 3, depths 1..8) and `worth_profile.py` (192
  rows) at each arm's LAST checkpoint: `mux_local` K1−K6 and K3−K6 with paired
  bootstrap CIs on X1; slot `ce_tokens` K1−K6 on X2/X3; plan worth.
- Val CE: the mean of the last four evals (eval every 250), never one eval.
- Wall clock per arm from the queue log against `to-mnext`'s 110 steps/min.

Spike step (pre-clip) := a probed step with `loop/cot_ratio` > 30.

## Predictions (frozen)

X1, stability:
- **P-X1a.** `to-mnext-ctt` reaches 5000 steps with the tripwire silent: 65%.
- **P-X1b.** The impulse is still there under the clip: at least 5 probed steps after
  step 1000 have `loop/cot_ratio` > 30 (pre-clip), AND the clip binds on them
  (`loop/cot_bind_max` > 0 on every such step): 70%.
- **P-X1c.** The clip costs less than 10 % wall clock against `to-mnext`'s 110
  steps/min at the same compiled+fused setting: 70%. (If the hook forces the compiled
  path to fall back, the smoke says so before launch and the arm runs eager; then this
  prediction is scored FALSE and the cost is reported.)

X1, the loop under the clip:
- **P-X1d.** The gain drift continues: `jac/rms_t3` on the fixed batch at X1's last
  checkpoint is ≥ 0.95 (C1 read 0.914 at 1150 and 1.000 at 1200; C0 0.889 flat): 55%.
- **P-X1e.** X1 THINKS by the panel rule: `mux_local` K1−K6 > 0.02 with the CI above 0
  AND K3−K6 > 0.01 at its last checkpoint: 35%. (`to-mnext` at its pre-onset 2500 read
  K1−K6 +0.0123 [+0.0104, +0.0142].)
- **P-X1f.** X1's last-four-eval val CE at 5000 is below `to-mown`'s single eval of
  4.2631: 50%.

X2/X3, the A1 pair at full BPTT:
- **P-X2a.** At least one seed shows the forecast-style spike train (≥ 3 spike steps
  after 1000): 60%.
- **P-X2b.** At least one seed trips: 55%. Both trip: 30%.
- **P-X2c.** On each seed's last checkpoint, slot `ce_tokens` K1−K6 > 0.02 with the CI
  above 0: 25%. (Every A1 reading in the record is ≤ 0.015.)
- **P-X2d.** The product on the healthy stretch (steps 1000–2000, non-spike steps) has
  median `loop/cot_ratio` > 3 on both seeds (8 iterations against A1-bptt4's 1.4–1.7 over
  4): 65%.

## Decision rule (binding)

- P-X1a TRUE and P-X1b TRUE ⇒ the clip is the stability lever for the forecast face and
  the impulse was the loop's product. Then P-X1e decides the next arm: TRUE ⇒ a 20k
  prereg of X1 against `notul-20k-wu` and a 20k R0; FALSE ⇒ phase 2 (a rate held below
  1 on the map, candidates (a)/(b) of the design note) is the next arm, with the clip kept.
- P-X1a FALSE ⇒ read the trip: if `loop/cot_post_ratio` ≤ 4 on every step of the last 50
  before the trip and `preclip/core` still crossed 1e4, the blow-up does not travel
  through the loop's product and §D's reading is amended (the weight path carries it);
  phase 2 is next and the clip is retired. If the trip coincides with `cot_post_ratio`
  > 4 (the reference itself exploded), the cap is relative to the wrong quantity and the
  next arm caps the exit cotangent as well.
- P-X2b TRUE with P-X2a TRUE ⇒ full BPTT is what exposes A1 to the product; the bptt-4
  A1 record is a truncation artefact on the stability axis. P-X2c stands alone: the
  contribution question is scored whatever the stability outcome.

## Method

Runner `/home/wolfe/morph-scratch/to/run_phase1.sh` (waits for the frozen-prereg flag
and for `run_phase1_prep.sh`; draws X1, X2, X3 in that order through the panel's v2
draw function with the tripwire watcher; then the depth sweeps and worth profiles at each
arm's last checkpoint; one trainer at a time). Prep: 12-step smokes of X1 and of the
unclipped twin at the compiled+fused setting (exit code, NaN lines, the `loop/cot_*`
probe keys, wall clock); `lab/divergence/cot_calibrate.py` on `ROLL_step_1150` with
four calm and four spike batches from the capture (the per-row product the clip sees).
Artifacts to `lab/experiments/results/2026-09-04-tul-clip-through-time/`. Estimated:
X1 ≈ 50 min, X2+X3 ≈ 70 min, readouts ≈ 45 min, sweeps ≈ 15 min.

## Not verified before launch

The hook under torch.compile at the real shape (CPU tests only, eager: 8 new + 62 TUL
tests pass at `4d5a986`); the per-row cap against the per-row calm product (the in-run
capture logged the global norm; the calibration in Method reads the per-row numbers and
is recorded here before launch); the reference's behaviour when the exit cotangent itself
spikes (the second branch of the decision rule); wandb rows for the clip keys (the local
`probe.jsonl` is the record either way).

### Method note, 2026-09-04 09:05 (prep results, before launch; Predictions untouched)

Smokes at the compiled+fused panel setting: X1 exit 0, 29 s for 12 steps, 0 NaN lines,
all six `loop/cot_*` probe keys present; the unclipped twin exit 0, 27 s, identical
step-12 loss (the cap did not bind in 12 steps). The hook survives torch.compile at this
shape. Calibration (`to/calibrate.log`; `ROLL_step_1150`, one labelled forward+backward per
captured batch, per-row cotangent over that row's exit cotangent, iteration 0 is the
largest): calm batches 1155/1159/1184 max row 2.04/1.37/1.93 (median rows 1.45/1.17/1.24);
batch 1194 (calm by `preclip/total` 3.5) one row at 6.28, median 1.09; spike batches
1161/1170/1185/1198: max row 2.09/5.09/16.39/2.56. At these weights the products are 3–20x
smaller than the same steps read in-run (2436 at 1170, 201 at 1185): the spike is weights
AND batch, and 20 updates of drift separate the checkpoint from the onset. The cap 4.0
binds no calm row but one of 48 and binds on iterations 0–1 of every spike batch; it is
kept as configured. The ratios fall below 1 by iteration 4 on every batch because the last
iterations' outputs carry only the slots still active there while the exit carries all.
