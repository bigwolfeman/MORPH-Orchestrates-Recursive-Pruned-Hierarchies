# Planned: phase 2 — bound the slot loop's forward map (state renorm; typical-gain penalty)

Status: planned
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
