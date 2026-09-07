# Planned: ARC E7 — the block-loop (the E4 mask at a deep slot draw, gradient through 8)

Status: planned
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
