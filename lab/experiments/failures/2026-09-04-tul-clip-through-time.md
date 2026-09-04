# Failure: clip-through-time on the forecast arm, and the A1 pair at full BPTT

Status: failure
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

### Method note, 2026-09-04 09:50 (during the run; Predictions untouched)

X1 tripped at 2764 (scored in Results). X2 (`to-a1-b8-s1`) tripped at 1682, and its
probe file carries NO `loop/cot_*` rows: `tul_to_a1_b8` inherits `training.loop_cot_probe:
false` from `base.yaml` and the prereg's spike-step definition reads that key. A method
error, mine. X3 was killed at its start (step < 200) and relaunched by
`run_phase1b.sh` with `training.loop_cot_probe=true` and nothing else changed, so one
seed carries the in-run readout. P-X2a and P-X2d are scored on X3 in-run; on X2 the
spike-train shape is read from `preclip/total` alone (single steps that snap back and
escalate, against the takeover's monotone ramp) and the product from the fixed-batch
Jacobian sweep over its 500-step checkpoints, and the record says so.

## Results (2026-09-04 10:20; X1 09:28, X2 09:42, X3 09:55; readouts and sweep 10:14; files in `results/2026-09-04-tul-clip-through-time/`)

| arm | verdict | onset of spikes | first `preclip/total` > 1e3 | last checkpoint | pace |
|---|---|---|---|---|---|
| X1 `to-mnext-ctt` | DETONATED 2764 | 1736 (pre-clip product > 30; bound on every step after) | 2764 | 2500 | 146 steps/min (unclipped fused draw: 110) |
| X2 `to-a1-b8-s1` | DETONATED 1682 | ≈ 1600 by `preclip/total` shape (no cot rows, Method note) | 1682 | 1500 | 150 steps/min |
| X3 `to-a1-b8-s2` | DETONATED 1144 | 1066 (68 spike steps by 1169) | 1140 | 1000 | 150 steps/min |

**X1, what the clip did and did not do** (200-step medians from `probe_to-mnext-ctt.jsonl`):

| window | pre-clip product t0 | carried after the clip | exit cotangent (`cot_norm_t7`) | `preclip/core` | `preclip/prelude` | exit state norm | iteration-0 realised gain | step ratio iter 3 / 7 |
|---|---|---|---|---|---|---|---|---|
| 1000–1199 | 72 | 72 | 33 | 0.3 | 1.1 | 1017 | 1.99 | 0.48 / 0.43 |
| 1600–1799 | 175 | 153 | 33 | 1.0 | 1.7 | 1327 | 2.45 | 0.59 / 0.55 |
| 1800–1999 | 3,594 | 418 | 46 | 33 | 26 | 1315 | 3.81 | 0.98 / 0.93 |
| 2200–2399 | 37,865 | 1,025 | 40 | 218 | 189 | 3349 | 8.86 | 1.14 / 1.05 |
| 2600–2799 | 36,636 | 512 | 27 | 253 | 294 | 4967 | 9.59 | 1.14 / 1.03 |

The cap bound on 100 % of rows on every step from 1736 (`loop/cot_bind_max` = 1.0),
the carried cotangent stayed at 4x the exit by construction, and the exit cotangent
never moved. The gradient grew anyway, on the weight path, while the forward inflated:
iteration 0's realised gain 1.5 → 9.6 over training and the trajectory's successive-step
ratio crossing 1.0 at 1800. Fixed-batch Jacobian sweep (`jac_sweep.jsonl`): `jac/rms_t3`
0.882 (500) → 0.896 (1000) → 0.894 (1500) → 1.885 (2000) → 8.663 (2500); `jac/rms_t0`
0.89 → 0.98 → 6.8 → 72.3; `jac/sigma_t0` 9 → 52 → 2,228 → 94,400. Under the clip the
map became expansive in its TYPICAL direction, by a factor the unclipped run never
reached before it died (C1 read 1.0 at its trip). Val CE moved first: 5.15 at 2250, 5.39
at 2500, 5.64 at 2750 against the unclipped draw's 4.65 at 2500; `mux_local` 6.92 → 7.37.

**X2/X3, the A1 pair at full BPTT.** Both detonated with the forecast face, not the
takeover ramp: `preclip/total` flat at 2.5–3.5 to within 100 steps of the onset, then
single steps at 37–750 (X2) and 242–1,242 (X3) snapping back to 3–14 between, the `tul`
family (E_slot / W_sent) among the largest, the exit cotangent flat (X3: 6 → 17) while
the product through the loop reached 1,123x at 1080 and 5,539x at 1150. X3's calm
product (steps 200–999) had median 2.43 and p90 2.82; its non-spike median after 1000
was 3.21. The sweep on their checkpoints: `jac/rms_t3` 0.89–0.91 on both, `jac/rms_t0`
0.91 → 0.94 → 0.999 on X2 (1500, 180 steps before its onset), step ratios rising 0.43 →
0.72 / 0.85 on X2 by 1500.

**Readouts at the last checkpoints** (480 rows; every checkpoint is inside or just before
the spike regime, so these are not 5000-step readings):

| arm | ckpt | token CE, depth 6 | slot / token K1−K6 | own-loss K1−K6 | own-loss K3−K6 | plan worth, offset 0 |
|---|---|---|---|---|---|---|
| X1 | 2500 | 5.4051 | +0.0087 [+0.0080, +0.0094] | +0.4185 [+0.3986, +0.4378] | +0.0932 [+0.0818, +0.1055] | (profile file) |
| X2 | 1500 | — | +0.0001 [−0.0001, +0.0002] | — | — | (profile file) |
| X3 | 1000 | — | +0.0001 [−0.0000, +0.0002] | — | — | (profile file) |
| `to-mnext` (unclipped) | 2500 | 4.6584 | +0.0006 | +0.0123 | −0.0004 | +0.070 |

X1's depth curve is steep on a DEGRADED model: its forecast loss at depth 8 (7.357) is
worse than the unclipped arm's at depth 1 (6.867), and its token CE is 0.75 nats worse at
the same step. The depth effect measures how far a one-iteration state sits from what a
head trained on an inflating eight-iteration state expects; it is not the loop earning.

Scored:

| prediction | credence | verdict |
|---|---|---|
| P-X1a X1 silent to 5000 | 65% | FALSE (2764) |
| P-X1b impulse still there and bound | 70% | TRUE (1,015 spike steps after 1000, bound on all) |
| P-X1c clip costs < 10 % wall clock | 70% | TRUE (146 vs 110 steps/min; smoke 29 s vs 27 s) |
| P-X1d `jac/rms_t3` ≥ 0.95 at the last checkpoint | 55% | TRUE (8.66; 1.89 at 2000) |
| P-X1e X1 THINKS by the panel rule | 35% | TRUE by the letter (0.42 / 0.09) on a model 0.75 nats worse than its control; not a THINK in substance, and it does not PAY |
| P-X1f val CE below `to-mown` at 5000 | 50% | unscorable (no 5000) |
| P-X2a ≥ 1 seed shows the spike train | 60% | TRUE (X3 by the metric, 68 spike steps; X2 by shape) |
| P-X2b ≥ 1 seed trips / both trip | 55% / 30% | TRUE / TRUE |
| P-X2c slot K1−K6 > 0.02 on the last checkpoints | 25% | FALSE on both (+0.0001) |
| P-X2d healthy-stretch product median > 3 on both | 65% | TRUE on X3 (3.21); X2 has no in-run rows and its checkpoint sweep reads 1.9–2.2 on the fixed batch before its onset — FALSE there |

## Verdict

Decision-rule branch 2: the blow-up does not need the loop's carried product. With that
product bounded at 4x the exit, the weight-path gradient grew 800x on an inflating
forward, and the map's typical gain went from 0.89 to 8.7. Clip-through-time is retired
as a cure and kept as an instrument (`loop/cot_post_*`, `loop/cot_bind_*`). Section D of
the divergence README is amended: the backward product is the first symptom; the disease
is the map moving past the edge, and bounding either half alone leaves the other to carry
it. On A1, full BPTT reproduces the same face on the token-CE target (both seeds, onsets
1066 and ≈ 1600): the bptt-4 A1 draws were quiet because the product was cut at four
iterations, and their loop earned nothing either way (P-X2c). The forecast target is not
what selects the face; eight iterations of gradient through the shared loop under
ternary on the LR plateau is.

## Updated hypothesis

The regime, not the gradient, has to be held: the next arms bound the forward map itself
(`2026-09-04-tul-forward-levers.md`: a per-slot state renorm between iterations, and a
hinge penalty on the map's typical gain measured every step). If both fail with the gain
held under 0.95, the alignment of the worst direction across the six shared blocks is the
remaining suspect and the deep-slot-stack arm (no weight sharing on the slot path) is
unparked.

## What the method could not distinguish

Whether an unclipped forecast arm's typical gain would also have reached 8 had it lived
to 2500 (C1 died at 1.0); whether a cap relative to the previous iteration instead of the
exit would behave differently (not run; the weight-path growth argues no); the A1 pair's
contribution at 5000 (no seed survived).
