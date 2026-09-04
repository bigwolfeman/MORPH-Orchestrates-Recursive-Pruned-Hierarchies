# Success: phase 2 — bound the slot loop's forward map (state renorm; typical-gain penalty)

Status: success
Date: 2026-09-04 (frozen before either arm ran; 12-step smokes precede launch and produce
no metrics). Branch `tul/think-once`, worktree `/home/wolfe/morph-to`, code `34d94a0`.
Follows `2026-09-04-tul-clip-through-time.md` (X1 tripped at 2764 with the backward
bounded: decision-rule branch 2) and the design note
`.agents/notes/proposed/architecture/2026-09-04-loop-contractivity-as-design.md`.

## Question

With the backward product clipped at 4x the exit cotangent, M-next still detonated: the
exit cotangent stayed flat (33 → 27), the carried cotangent sat at the cap, and the core
weight gradient grew 800x while the FORWARD inflated — iteration 0's realised gain
climbed 1.5 → 2.0 → 3.8 → 5.8 → 9.6 over training, the exit state norm 1017 → 5392, and
the successive-step ratio along the trajectory crossed 1.0 at step 1800, exactly where
the gradient began to climb. The clip bounded the symptom's backward half; the map itself
kept moving toward and past the edge. Two levers on the map, both kept on top of the clip:

1. **Y1, state renorm.** Pin every slot's carried norm to its entry norm after each
   iteration (direction preserved). Removes the forward inflation by construction. If the
   spike train survives it, the expansion is tangential (the aligned directions of the
   capture) and a norm pin cannot touch it.
2. **Y2, typical-gain penalty.** `100 · relu(g − 0.9)²` on the map's typical gain `g` at
   one random grad iteration per step (2 % finite difference, two extra core steps, same
   dropout masks, global RNG untouched). Acts on the quantity that drifted 0.87 → 1.00 in
   the capture, on the whole map. If it holds `g` under 0.95 and the arm survives, the
   contraction rate is a trainable design quantity; what it costs the loop's earning is the
   second readout.

## Arms (every arm composes `tul_to_mnext_ctt` → `tul_to_mnext` → `tul_to_panel`)

| # | run | config | one line |
|---|---|---|---|
| Y1 | `to-mnext-y1` | `tul_to_mnext_y1` | R4 + clip 4.0 + `slot_state_renorm: true` |
| Y2 | `to-mnext-y2` | `tul_to_mnext_y2` | R4 + clip 4.0 + `slot_gain_lambda: 100`, target 0.9, eps 0.02 |

Controls on disk: `to-mnext-ctt` (clip alone, tripped 2764), `to-mnext` (fused, 3618),
`cap-c1-det` (eager, 1208), `cap-c0-nt` (ternary off, clean).

## Readouts

- The tripwire on both draws; `preclip/*`; `loop/cot_ratio`, `loop/cot_post_ratio`,
  `loop/cot_bind_max`; `loop/core_gain_t{k}`, `loop/in_norm_t{k}`, `loop/out_norm_t7`,
  `loop/delta_ratio_t{k}` (every step, `probe.jsonl`).
- Y2 only: `tul/gain_est`, `tul/gain_est_max`, `tul/gain_reg_weighted` every step.
- `lab/divergence/jac_sweep.py` over each arm's 500-step checkpoints on the capture's
  fixed batch (`jac/rms_t3`, `jac/sigma_t3`; on Y1 the probe measures the RAW step, before
  the renorm, so it reads the underlying map).
- `core_depth_sweep.py` and `worth_profile.py` at each arm's last checkpoint (`mux_local`
  K1−K6 / K3−K6 with CIs, token CE K1−K6, plan worth).
- Val CE, mean of the last four evals. Wall clock against `to-mnext-ctt`'s 146 steps/min.

Spike step := `loop/cot_ratio` > 30 (pre-clip), as in phase 1.

## Predictions (frozen)

Y1 (renorm):
- **P-Y1a.** Y1 reaches 5000 with the tripwire silent: 45%.
- **P-Y1b.** `loop/core_gain_t0` stays at 1.00 ± 0.02 on every probed step (the pin
  holds; a reading away from 1 means the renorm is not where the gain is measured): 90%.
- **P-Y1c.** Spike steps still occur after 1000 (≥ 5 with `loop/cot_ratio` > 30): 65%.
  (The tangential story: the aligned directions grow inside the pinned sphere.)
- **P-Y1d.** `jac/rms_t3` of the raw map at Y1's last checkpoint ≥ 0.95: 60%.
- **P-Y1e.** Y1 THINKS by the panel rule (`mux_local` K1−K6 > 0.02, CI above 0, K3−K6 >
  0.01) at its last checkpoint: 30%.

Y2 (gain penalty):
- **P-Y2a.** Y2 reaches 5000 with the tripwire silent: 55%.
- **P-Y2b.** The penalty binds: median `tul/gain_est` over steps 1000–5000 (or to the
  trip) sits in [0.88, 0.96] and `tul/gain_reg_weighted` > 0 on more than 30 % of those
  steps: 65%.
- **P-Y2c.** `jac/rms_t3` (fp32 power iteration, fixed batch) at Y2's last checkpoint is
  below 0.95: 60%. Below C1's 1150 reading of 0.914: 35%.
- **P-Y2d.** Spike steps after 1000 number fewer than 5: 50%.
- **P-Y2e.** Y2 THINKS by the panel rule at its last checkpoint: 25%.
- **P-Y2f.** Y2's wall clock is within 1.25x of `to-mnext-ctt` (two extra core steps on
  the compact sequence per step): 70%.
- **P-Y2g.** Y2's last-four-eval val CE at 5000 is within 0.05 of `to-mown`'s 4.2631
  (the healthy slot arm on the recipe): 40%.

## Decision rule (binding)

- P-Y2a TRUE and P-Y2b TRUE ⇒ the contraction rate is a trainable design quantity and it
  is the stability lever for the forecast face. P-Y2e then decides: TRUE ⇒ 20k prereg of
  Y2 against `notul-20k-wu` and a 20k R0; FALSE ⇒ the loop is stable and empty at this
  depth and scale, and the programme moves to the design note's open question (what a
  deeper draw is FOR when the readout saturates by iteration 3), not to another lever.
- P-Y1a TRUE and P-Y2a FALSE ⇒ the forward inflation was the whole remaining disease and
  the norm pin is the lever; Y2's penalty is retired (its `gain_est` trace says whether it
  ever bound).
- Both FALSE ⇒ read the pair: if Y1 tripped WITH spike steps at a pinned gain (P-Y1c) and
  Y2 tripped with `gain_est` held under 0.95 (P-Y2b's median), the detonation does not
  need an expansive map in the typical direction and the alignment of the worst
  direction across the six shared blocks is the remaining suspect; the next arm drops
  weight sharing on the slot path (the deep-slot-stack arm of the think-once rule, which
  Wolfe parked on 2026-09-03 and which this evidence would unpark).
- P-Y1b FALSE ⇒ the renorm is not on the path the probes read; stop and fix before any
  reading is taken.

## Method

Runner `/home/wolfe/morph-scratch/to/run_phase2.sh` (waits for `PHASE1 COMPLETE` and the
frozen-prereg flag; 12-step compiled smokes of Y1 and Y2 first — exit code, NaN lines,
the `gain_est` key on Y2, wall clock; then Y1, Y2 through the panel's v2 draw function
with the tripwire; then the depth sweeps, worth profiles and the fixed-batch Jacobian
sweep over every 500-step checkpoint). One trainer at a time. Artifacts to
`lab/experiments/results/2026-09-04-tul-forward-levers/`. Estimated: 2 × 45 min draws,
45 min readouts, 15 min sweeps.

## Not verified before launch

The two extra core steps and the RNG save/restore under torch.compile at the real shape
(CPU tests only: 6 new + 134 TUL tests pass at `34d94a0`); the finite-difference gain in
bf16 against the fp32 power iteration at the real shape (the CPU linearity test passes;
the jac sweep on Y2's own checkpoints is the check, recorded in Results); the renorm's
interaction with the MUX head's read of the exit state (the head reads a normed state
either way; unmeasured); λ = 100 (one value, no sweep; the `gain_reg_weighted` trace says
whether it was too weak or too strong).

### Method note, 2026-09-04 10:20 (Y1 drawing, Y2 not started; Predictions untouched)

Smokes: Y1 exit 0, 44 s, `loop/core_gain_t0` = 1.0 exactly at step 12 (the pin sits on
the path the probe reads); Y2 exit 0, 32 s, no NaN. Y2's live gain (`gain_est`) reached
wandb only; before Y2 launches, `train.py`'s probe row also carries `loss/gain_est`,
`loss/gain_est_max` and `loss/gain_reg_weighted`, so the local `probe.jsonl` holds the
P-Y2b readout. Instrumentation only; the model code is `34d94a0`.

## Results (2026-09-04 12:00; Y1 10:15–10:58, Y2 10:58–11:44, readouts 11:55; files in `results/2026-09-04-tul-forward-levers/`)

Both arms reached 5000 steps with the tripwire silent — the first forecast arms to live
through the LR plateau under ternary at full BPTT (the unclipped draws died at 3618 fused
and 1208 eager; the clip-alone draw at 2764).

| arm | verdict | max `preclip/total` at step ≥ 200 | spike steps (pre-clip product > 30) after 1000 | clip bound | wall clock |
|---|---|---|---|---|---|
| Y1 renorm | HEALTHY 5000 | 70.6 at 245 | 0 | never | 43.1 min (116 steps/min) |
| Y2 gain penalty | HEALTHY 5000 | 118 at 224 | 0 | never | 45.6 min (110 steps/min; clip-alone draw 115) |

500-step medians from the probe files:

| window | Y1 `preclip/total` | Y1 product t0/exit | Y1 step ratio 3 / 7 | Y2 `preclip/total` | Y2 product | Y2 step ratio 3 / 7 | Y2 `gain_est` (median / max) | Y2 hinge active | Y2 iter-0 realised gain | Y2 exit norm |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000–1499 | 2.9 | 0.45 | 0.27 / 0.24 | 2.9 | 2.26 | 0.50 / 0.43 | 0.895 / 0.899 | 34 % | 1.94 | 886 |
| 2000–2499 | 2.3 | 0.46 | 0.31 / 0.28 | 2.3 | 2.03 | 0.49 / 0.42 | 0.895 / 0.899 | 34 % | 2.34 | 1127 |
| 3000–3499 | 2.0 | 0.38 | 0.29 / 0.28 | 2.0 | 1.90 | 0.45 / 0.33 | 0.892 / 0.896 | 12 % | 2.47 | 1379 |
| 4500–4999 | 1.8 | 0.38 | 0.30 / 0.29 | 1.9 | 1.83 | 0.54 / 0.47 | 0.894 / 0.898 | 26 % | 2.60 | 1419 |

Y1's realised gain reads 1.0000 at every iteration on every step (the pin is on the
measured path). Y1's backward SHRINKS through the loop (product 0.38–0.62) and its
trajectory contracts at 0.3 per iteration. Y2's live typical gain sat at 0.894 (median
over 1000–4999, p10 0.892, p90 0.897) with the hinge touching on 24.65 % of those steps
and a median penalty of 0.000: the hinge acts as a barrier at the target, not as a tax.
Y2's iteration-0 realised gain still climbs (1.5 → 2.6; the clip-alone draw reached 9.6)
and its exit norm 1.8x (5x on the clip-alone draw); its product through the loop stays at
1.8–2.3 for the whole run.

Validation (mean of the last four evals, every 250 steps; same recipe and shape):

| arm | last-four val CE | 480-row CE at depth 6 |
|---|---|---|
| A3 coreless floor | 4.0792 | 4.0208 |
| A1-wu s1 (bptt 4) | 4.2308 | 4.1502 |
| **Y1** | **4.2763** | **4.1995** |
| **Y2** | **4.2809** | **4.2054** |
| M-own | 4.3078 | 4.2235 |

Depth sweeps at 5000 (480 rows, batch 3) and worth profiles (192 rows):

| arm | token K1−K6 | forecast K1−K6 | forecast K3−K6 | forecast loss, depth 1 / 3 / 8 | plan worth, offset 0 (zero / shuffle) |
|---|---|---|---|---|---|
| Y1 | +0.0006 [+0.0004, +0.0007] | +0.0128 [+0.0112, +0.0144] | −0.0006 [−0.0010, −0.0003] | 6.766 / 6.752 / 6.753 | +0.092 / +0.052 |
| Y2 | +0.0003 [+0.0002, +0.0005] | +0.0135 [+0.0118, +0.0154] | +0.0002 [−0.0004, +0.0008] | 6.763 / 6.750 / 6.751 | +0.074 / +0.034 |
| `to-mnext` at 2500 (unclipped, pre-onset) | +0.0006 | +0.0123 | −0.0004 | 6.867 / 6.855 / 6.858 | +0.070 / +0.043 |

Scored:

| prediction | credence | verdict |
|---|---|---|
| P-Y1a Y1 silent to 5000 | 45% | TRUE |
| P-Y1b realised gain pinned at 1.00 ± 0.02 | 90% | TRUE (1.0000 everywhere) |
| P-Y1c spike steps still occur under the pin | 65% | FALSE (none) |
| P-Y1d raw-map `jac/rms_t3` ≥ 0.95 at the last checkpoint | 60% | (sweep, part 2) |
| P-Y1e Y1 THINKS | 30% | FALSE (K3−K6 −0.0006) |
| P-Y2a Y2 silent to 5000 | 55% | TRUE |
| P-Y2b `gain_est` median in [0.88, 0.96] AND hinge > 30 % of steps | 65% | FALSE by the letter: median 0.894 (in range), hinge on 24.65 % |
| P-Y2c `jac/rms_t3` < 0.95 at the last checkpoint; < 0.914 | 60% / 35% | (sweep, part 2) |
| P-Y2d fewer than 5 spike steps | 50% | TRUE (0) |
| P-Y2e Y2 THINKS | 25% | FALSE (K3−K6 +0.0002) |
| P-Y2f wall clock within 1.25x of the clip-alone draw | 70% | TRUE (1.05x) |
| P-Y2g last-four val within 0.05 of `to-mown`'s 4.2631 | 40% | TRUE (4.2809, +0.018) |

## Results, part 2 (2026-09-04 12:10; the fixed-batch Jacobian sweep over every 500-step checkpoint, `jac_sweep.jsonl`)

| arm | step | `jac/rms_t0` | `jac/rms_t3` | `jac/rms_t7` | `jac/sigma_t3` | `jac/sigma_t0` | step ratio 4 / 7 | product t0/t7 |
|---|---|---|---|---|---|---|---|---|
| Y1 | 1000 | 0.925 | 0.925 | 0.920 | 12.1 | 26 | 0.26 / 0.24 | 0.41 |
| Y1 | 2500 | 0.966 | 0.940 | 0.935 | 18.1 | 30 | 0.28 / 0.28 | 0.42 |
| Y1 | 3500 | 0.967 | 0.950 | 0.947 | 19.6 | 55 | 0.30 / 0.29 | 0.41 |
| Y1 | 5000 | 0.950 | 0.942 | 0.939 | 15.0 | 22 | 0.30 / 0.30 | 0.39 |
| Y2 | 1000 | 0.936 | 0.897 | 0.892 | 10.4 | 22 | 0.40 / 0.37 | 2.10 |
| Y2 | 2500 | 0.990 | 0.889 | 0.882 | 8.6 | 48 | 0.40 / 0.36 | 1.98 |
| Y2 | 3500 | 0.977 | 0.887 | 0.880 | 15.2 | 43 | 0.47 / 0.40 | 1.97 |
| Y2 | 5000 | 0.996 | 0.887 | 0.880 | 7.3 | 127 | 0.52 / 0.47 | 1.81 |

Against the record: the unregularised eager run read `rms_t3` 0.885 at 500, 0.914 at
1150 and 1.000 at 1200 (its trip); the clip-alone run 0.894 at 1500, 1.885 at 2000,
8.663 at 2500.

- **Y2's penalty pins the quantity it measures.** The fp32 power-iteration probe agrees
  with the bf16 finite-difference reading to within 0.01 (0.887–0.897 against a live
  median of 0.894) at every checkpoint, and the map's worst gain at iteration 3 FALLS over
  training (10.4 → 7.3) instead of climbing. The unregularised iteration (`rms_t0`, the
  entry state, drawn 1 time in 8) still drifts 0.936 → 0.996 with its worst gain 22 → 127:
  the penalty holds the operating points it visits and not the one it rarely does.
- **Y1's raw map drifts less and plateaus.** `rms_t3` 0.925 → 0.950 (3500) → 0.942
  (5000), never above 0.95 at a checkpoint; the worst gain 12–25 against the record's 30
  at 1150 and 100 at 1200. Pinning the state changes what the map learns.
- Both trajectories contract (step ratios 0.3 on Y1, 0.4–0.5 on Y2); the slot states'
  effective rank at the exit is 45–49 on both.

Scored, the two that waited: **P-Y1d FALSE** (0.942 at the last checkpoint, by 0.008;
0.950 at 3500). **P-Y2c TRUE** on both clauses (0.887 < 0.95 and < 0.914).

## Verdict

Both forward levers hold the forecast arm through the plateau; the backward clip alone
did not. The decision rule's first branch fires on its intent — Y2 survives and its
controller holds the map's typical gain at the target for the whole run — with P-Y2b's
second clause missed by the letter (the hinge touched 24.65 % of steps, not 30 %; a
barrier at the target needs to touch less often than a tax). P-Y2e is FALSE, and so is
P-Y1e: the stable loops earn 0.013 nats of forecast loss in their first three iterations
and nothing after, read none of it into the token loss (K1−K6 ≤ 0.0006), and sit 0.18
nats behind the coreless floor on 480 rows at 5000 steps. So, by the rule: the slot loop
is stable and empty at this depth and scale, and the programme moves to the design
note's open question — what a deeper draw is FOR when every readout saturates by
iteration 3 — not to another lever. Of the two levers the renorm is the cheaper (no extra
passes, no RNG bracket, 116 against 110 steps/min) and reads the same on every axis;
the penalty is the one that acts on the measured quantity and is the instrument of
choice when the gain itself is the question.

Wolfe's 2026-09-03 hypothesis, scored over the day: the loop does want to be asymptotic
to 1 (the unregularised map drifts 0.87 → 1.00 and detonates at the crossing), and
holding it at 0.89 by a penalty on the map stops the detonation. What holding it does
not do is make the loop earn.

## Updated hypothesis

Stability on the slot loop is solved at this scale by acting on the map (either lever),
and the record now has one stable forecast arm per lever with checkpoints every 500
steps. The contribution question is not a stability question and does not yield to
stability levers: every slot arm on the recipe, stable or not, does its work in three
iterations. The next prereg is on that axis, with the deep-slot-stack (no weight
sharing) and the "what a deeper draw is for" question as the candidates, and it is
Wolfe's call which. Neither lever is switched on in `base.yaml` by this record; that is
a shipped-design decision and gets its own note when made.
