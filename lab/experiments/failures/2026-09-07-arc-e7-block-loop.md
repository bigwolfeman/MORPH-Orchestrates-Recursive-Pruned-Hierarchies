# Failure: ARC E7 — the block-loop (the E4 mask at a deep slot draw, gradient through 8)

Status: failure
Date: 2026-09-07 (frozen before any smoke; Wolfe's call: "test it")
Arc: `2026-09-04-loop-contribution-arc.md`, Method amendment 3. Design 1 of the 2026-09-07
design list; follows `failures/2026-09-04-huginn-loop-contribution.md` and E6
(`planned/2026-09-07-arc-e6-deep-recurrence-draw.md`).

## Question

A loop earns only what its loss demands (the slot loop's next-token target was empty;
the paid loop earns because every position carries the full NTP loss) and, on Huginn,
it earns up to roughly half its training depth. The block-loop puts the whole next span's
CE on the slot by making the slot the only cross-span route (E4's mask), and trains the
slot loop at a deep draw with a short gradient window (E6's regime). Does a slot that is
load-bearing AND deep earn past iteration 3, and do the tokens read that depth?

## Method

`morph/configs/tul_to_mnext_y2_mask_d16.yaml`: `tul_to_mnext_y2_mask` (Y2 + `tg_restrict`,
eager attention) with `tul.slot_mean_depth 16`, `tul.slot_max_depth 24`, `model.bptt_depth
8`. Hinge unchanged (0.90, lambda 100, single random grad iteration), cot clip 4, ramp
1000, seq 1024, batch 6, 5000 steps, seed 1, `ademamix_alpha_cap 3.5` (panel pin).
Sustained tripwire on `probe.jsonl`. At 2500 and 5000: `core_depth_sweep.py` at forced
slot depths 1, 3, 6, 8, 12, 16, 24 on the arc's 480 rows (token CE and forecast
`mux_local`, paired CIs); `worth_profile.py` on 192 rows. Rulers: Y2 at 5000 (forecast
K1−K6 +0.0135, K3−K6 +0.0002, val 4.2809), E4 if it has run, E6 for the plain regime.
Output `results/2026-09-07-arc-e7/`. Runs AFTER E6 in the same queue
(`arc/run_e6_e7.sh`).

Cost, not measured: the slot loop runs on the compact slot sequence (~1/8 of positions),
so 16 iterations there cost about what 2 token iterations would; the mask's eager
attention was the R5 arm's cost (~1.5 s/step at batch 6). ~2 h for 5000 steps.

## Predictions (frozen)

- **P7a.** Reaches 5000 with the sustained tripwire silent: **50 %** (a 24-iteration
  slot map with 16 no-grad iterations in front of the hinge is untested).
- **P7b (THINK bar).** Forecast `mux_local` K3−K6 at 5000 > 0.01 nats with the paired CI
  above 0: **35 %** (every stable slot arm ≤ 0.0014; the depth and the route both change).
- **P7c (the reader).** Token K1−K6 at the slot ≥ 0.01 (E4's P4d bar): **30 %**.
- **P7d.** Token K6−K12 at the slot > 0.005 with CI above 0: **20 %**.
- **P7e (the price).** Last-four val CE within 0.15 of Y2's 4.2809: **40 %** (the mask
  cost the gist family 0.09–0.36; the deep draw converges slower).
- **P7f.** The map's gain estimate (`loss/gain_est`, grad iterations only) sits above
  0.90 in more than 20 % of steps after 2000 (the hinge active: near-fixed-point
  iterations read closer to 1 than Y2's 0.89): **50 %**.

## Binding

- P7b or P7c TRUE ⇒ the block-loop is the TUL design: E5 runs on it at 20k, matched wall
  clock against notul-20k and A2-20k, scored on val CE per second of training and on
  the same sweeps.
- P7b and P7c FALSE, P7a TRUE, and E6's P6a TRUE ⇒ the regime earns on the plain loop
  but not on a load-bearing deep slot: the slot's problem is neither depth nor route.
  Design 2 (per-position depth on the paid loop) is next and needs code.
- E6's P6a FALSE ⇒ E7's negative is not readable (the regime did not earn on the plain
  loop either); E7's positive still stands on its own.
- P7a FALSE ⇒ the deep slot draw is a stability experiment first; file under the
  divergence README's regime table with the probe record.

## Not verified before launch

Memory and step time with `slot_max_depth 24` under eager attention (no smoke has run);
the hinge's single-iteration draw under 16 no-grad iterations (it draws from the grad
window only); `core_depth_sweep.py` at forced depths above the model's 24 (it raises
`slot_max_depth` per call; the 24 cell is new); whether `W_prefix` and the prefix
positions behave at a per-slot depth spread of 1..24 (they read the FINAL state
regardless of depth; the spread only changes the trajectory length).

## Results (2026-09-07 02:31; `arc/run_e6_e7.sh`, worktree e0501b2; draw 01:36–02:23, DETONATED at 2712; files in `results/2026-09-07-arc-e7/`)

Training: 1.17 steps/s, peak 19.4 GB. The sustained tripwire fired at step 2712 (rows
> 1e4 at 2684 and 2712, max 5.68e4). Probe record (`probe_to-mnext-y2-mask-d16.jsonl`):

| step | preclip/total | gain estimate (hinge) | hinge penalty (nats) | loop state norm, iteration 7 |
|---|---|---|---|---|
| 500 | 6.3 | 0.892 | 0.00 | 1.5e3 |
| 1000 | 37.9 | 1.118 | 4.84 | 1.6e5 |
| 1143 | 740.6 | 1.186 | 8.47 | 1.4e6 |
| 2000 | 11.6 | 1.049 | 2.25 | 3.9e7 |
| 2500 | 22.9 | 1.038 | 1.97 | 1.4e8 |
| 2684 | 27430 | 1.265 | 45.1 | 1.6e8 |
| 2712 | 56757 | 1.279 | 45.2 | 1.7e8 |

The hinge lost: from step 1000 the sampled gain sat at 1.03–1.06 under a 2–5 nat penalty
and never returned below 0.9. `mux_local` stayed at 7.1–7.3 for the whole run. Val CE at
2500: 4.892 (Y2 at 2500: 4.634).

Pre-onset checkpoint at 2500, forced slot depth on the 480 rows (`sweep_*_2500.json`):

| depth | 1 | 3 | 6 | 8 | 12 | 16 | 24 |
|---|---|---|---|---|---|---|---|
| token CE | 4.918 | 4.894 | 4.892 | 4.892 | 4.892 | 4.892 | 4.892 |
| forecast `mux_local` | 8.075 | 7.280 | 7.252 | 7.239 | 7.232 | 7.232 | 7.232 |

Token K1−K6 +0.0263 [+0.0238, +0.0289], K3−K6 +0.0018 [+0.0016, +0.0022]; forecast K1−K6
+0.823, K3−K6 +0.0279 [+0.0188, +0.0373]. Rulers on the same rows: Y2 at 5000 token K1−K6
+0.0003, forecast K3−K6 +0.0002; the E2 pre-onset checkpoint at 2500 token K1−K6 +0.0019,
forecast K3−K6 +0.0077. Plan worth at 2500 (`worth_*_2500.json`, 192 rows): zero 0.056 at
offset 0, shuffle ≈ 0.000, wrong-seed 0.305.

Jacobian sweep on the 2500 checkpoint (`jac_sweep_e7.jsonl`, fp32 power iteration, 24
iterations captured; the healthy arc-a arms beside it):

| | E7 mask+deep @2500 | Y2-g95 @2500 | Y2-iter @2500 (pre-onset) |
|---|---|---|---|
| loop entry norm | 485 | 489 | 486 |
| state norm after iteration 0 | 1.5e7 | 1.2e3 | 1.8e3 |
| state norm after iteration 7 | 1.4e8 | 1.5e3 | 1.3e3 |
| realized gain, iteration 0 | 34847 | 2.5 | 4.0 |
| realized gain, iterations 1 → 23 | 2.24 → 1.02 | | |
| σ_max of the step map, iteration 3 / 7 / 15 | 95 / 12 / 5.7 | 22 / 15 | 87 / 30 |
| typical (rms) gain of the step map, iteration 3 / 7 / 15 | 0.86 / 0.85 / 0.85 | 0.92 / 0.91 | 0.97 / 0.93 |
| σ_max of block 5 alone, iteration 0 / 3 / 15 | 93864 / 74 / 13 | 6.5 (it. 3) | 7.7 (it. 3) |
| effective rank of the state, iteration 0 → 7 → 23 | 181 → 125 → 9 | 181 → 46 | 170 → 39 |
| per-iteration change ratio, iteration 3 → 23 | 0.53 → 0.03 | 0.74 | 0.50 |

Weights at 2500 against Y2 at 5000: block 5's MLP down shadow weight 96 vs 33 (norm), its
HC write projections 15 and 16 vs 4.0 and 4.3, its norm gains 1.33–1.39 vs 1.03 max,
`injection.log_dt` 0.18 vs 0.79, CCA alpha 0.59 vs 0.07. No weight is at 1e4; the 3e4x
first-iteration amplification is a property of the composed map, not of one tensor.

Scored:

- **P7a FALSE.** Detonated at 2712.
- **P7b, P7c: not scorable at 5000.** At the pre-onset 2500 checkpoint the forecast K3−K6
  (+0.028) and the token K1−K6 at the slot (+0.026) both clear their bars, on a map that
  had sat at gain 1.04 since step 1000 with σ_max 95. Per the E2 rule, a pre-onset
  reading is not a lever's earning. Fourth co-occurrence of earning and instability.
- **P7d: not scorable.** Token K6−K12 at 2500 is 0.0000: the slot loop saturates at 3
  even at a mean-16 draw. The draw is not the lever on the slot path (contrast E6).
- **P7e FALSE** at 2500 (+0.26 against Y2).
- **P7f TRUE** (100 % of steps after 1000).

## Verdict

The block-loop as configured is a stability experiment, and its Jacobian says what the
disease is in numbers. The map sends the 485-norm slot entry state to 1.5e7 in ONE
iteration (block 5's σ_max 9.4e4 at iteration 0), after which the carrier is so large that
each further iteration is a 2 % rotation: the loop becomes a power iteration, the state's
effective rank falls 181 → 9, the per-iteration change ratio falls to 0.03, and nothing
past iteration 3 changes the readout. Saturation at 3, state-norm growth and the
detonation are one mechanism. The hinge could not see it: it measures a random-direction
finite-difference gain (0.85–1.04) while the top singular direction sits at 95.

Two defects found while reading the run, both verified in code:

1. **The no-grad prefix is one global count per batch** (`n_nograd = total_iters −
   bptt_depth`, `transformer.py:1790` and `:2436`), and a sample whose depth is at or
   below it runs every iteration under `no_grad`: its loop output carries no gradient
   to the core or the prelude. Simulated on the draws used: 27 % of E6's samples, 57 %
   of E7's slots, and 25 % under the old `bptt_depth 4` recipe. Parcae front-pads so
   every sequence's LAST ⌈µ/2⌉ iterations carry gradient (Alg. 2). Note:
   `.agents/notes/proposed/bug-fix/2026-09-07-truncated-bptt-silences-shallow-samples.md`.
2. **The diagonal carry constraint covers 256 of 768 carrier dims** (`DiagonalInjection`
   on the context channel only, `transformer.py:409`); the remaining 512 ride the
   HC-Cayley residual, which is norm-preserving by design. Parcae's ρ(A) < 1 is on the
   whole residual and is the paper's entire stability claim (§4.1, Table 1, Fig. 3).

Not verified: which block or write inside iteration 0 produces the 3e4x jump (the
App. J decomposition, per-block state norms within one iteration, has not been run);
whether the mask alone, at mean 6, reads the slot's depth (E4 as preregistered never
ran); whether the 27 % silent-sample defect changes E6's numbers (its control notul at
full BPTT has no silent samples).

## Updated hypothesis

The slot loop's carrier is unbounded and its write is unbounded, so a deep draw lets one
iteration set the scale and the rest rotate. Bounding the STATE (a norm on the carrier
between iterations, `slot_state_renorm`, or a bounded write) is upstream of any gain
penalty, and the penalty itself must be directional (a JVP power-iteration term on the
top singular direction, STARS 2605.26733) to see a rank-few expansion. Before any of that,
the truncation defect must be fixed so a deep draw trains every sample. The next
stability run is the block-loop with the per-sample gradient window, `slot_state_renorm`
on, and the hinge kept, under the abort rule; it is a stability prereg, not a
contribution one, and it is Wolfe's call.
