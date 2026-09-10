# Planned: the slot-map levers — why the slot loop does not move its state

Status: planned
Date: 2026-09-10 (frozen before launch; Wolfe, 01:10, going to bed: "What arms should we
add on for the next 8 or so hours to finally solve this?"). Arc:
`2026-09-04-loop-contribution-arc.md`. Follows `2026-09-09-arc-coda-reads-the-thought.md`
(the unpack arm) and `2026-09-09-arc-slot-loop-norm-match.md`.

## Question

The unpack arm closed the coda's incentive as a suspect: with the coda's only route to
earlier spans being z, z is used (zeroing it costs 0.811 nats at offset 0, shuffling it
1.651) and the depth curve is still flat (tokens K1−K6 +0.0005, K3−K6 +0.0000; forecast
K1−K6 +0.0038 at 5,000). A slot looped six times gives the coda the same z as a slot looped
once. The slot-state probe on the no-MUX arm reads 1–3 % movement per pass. What pins the
slot map near the identity? Three levers the tree already has, one factor each on the
unpack arm:

1. **The stability terms.** The gain hinge (`slot_gain_lambda` 100 at 0.9) and the
   terminal fixed-point term (`core_fixed_point_lambda` 1.0) both reward a map whose
   output stops moving. They were added against the spike train, not measured for what
   they cost the loop's depth on this forward.
2. **The entry.** The slot state enters the loop AS the prelude state (`core_state_init
   prelude`, ctx-only injection). The plain core under Parcae's noise entry moved its
   state 61/28/16 % on its first passes; the entry confound was raised 2026-09-09 and is
   unmeasured on the slot loop.
3. **The depth draw.** The per-slot Poisson draw (mean 6, max 8) averages the loss over
   depths 1..8; a depth-invariant z is the safe answer. A fixed depth removes that reward.

Not run tonight, recorded as the structural hypothesis: inside `_tul_core` the compact
sequence holds only slot cells, so a pass has nothing new to read — the paid loop, the one
arm that earns depth, re-reads every token state every pass. The lever for it is a
cross-attention read of the frozen prelude token states from the looping slot (code, not
a knob); it is the next arm if the three above read flat.

## Hypothesis

H-lev-1: the stability terms are the pin; removing them lifts the per-pass movement above
5 % and K3−K6 above zero, at the price of the spike risk they were built against. H-lev-2:
the entry is the pin; a state that starts as noise must be built by the map, so the first
passes move by construction and the tokens read the difference. H-lev-3: the draw is the
pin; a fixed depth makes z depth-specific. H-lev-0 (the null): none of the three moves
K3−K6; the map on a slots-only sequence has nothing to compute per pass, and the reread
arm is next.

## Method

All arms are `tul_slot_unpack_norm_match` plus ONE change, on the think-once panel recipe
(seq 1024, batch 6, seed 1, 5,000 steps, ramp 1,000, `tg_scoped_kernels`), runner
`arc/run_slotloop3.sh` (file-driven queue, commit pinned per arm in
`arc/slotloop3_arms.txt`), per arm a 12-step smoke, the draw with the sustained tripwire
and the step-200 rate rule (skip the arm, continue), then the slot-loop readout (sweeps at
forced slot depth 1,2,3,6,9,12,16 at 2,500 and 5,000 with the token and forecast curves;
`worth_profile.py` at 5,000; the slot-state probe at 5,000). Order: the plain ruler
`plain-panel-norm-match` first (the prelude-entry plain core's anatomy under norm_match,
the comparison every reading below needs), then the three levers, then the deferred
`slot-mux-absmean` and the two recipe-reads arms.

| arm | config | the one change |
| --- | --- | --- |
| `slot-unpack-free` | `tul_slot_unpack_free` | `slot_gain_lambda 0`, `core_fixed_point_lambda 0` (`slot_cot_clip` 4.0 stays) |
| `slot-unpack-noise-entry` | `tul_slot_unpack_noise_entry` | `notul_parcae_entry`'s entry block: noise init, all-dim carry with learned B (constraint terms as shipped) |
| `slot-unpack-fixed-depth` | `tul_slot_unpack_fixed_depth` | `tul.slot_depth_fixed 6` |

Readers: `sweep_score.py` on the token and forecast curves against the unpack arm's own
readout (`results/2026-09-09-slot-loop-norm-match/`), `slot_state_probe.py` for the
movement. Results to `lab/experiments/results/2026-09-10-slot-map-levers/`. The
fixed-depth arm's sweep at depths other than 6 reads off-distribution sensitivity, not
earning; its own reading is CE at depth 6 against the unpack arm's at depth 6 on the same
480 rows, and the probe's movement.

## Predictions (frozen)

- **P-lev-a (survival).** HEALTHY to 5,000: free **60 %** (the spike train the terms were
  built against), noise-entry **75 %**, fixed-depth **55 %** (E13's scale mode).
- **P-lev-b (free).** Slot-state movement per pass (probe, passes 2–6 mean) above 5 %:
  **45 %**. Tokens K1−K6 above 0.005 with the CI above 0: **35 %**; K3−K6 above 0.002:
  **25 %**. Forecast K1−K6 above 0.010: **40 %**.
- **P-lev-c (noise entry).** First-pass movement above 30 %: **90 %** (by construction).
  Tokens K1−K6 above 0.005: **30 %**; K3−K6 above 0.002: **25 %**. Val CE at 5,000 within
  0.05 of the unpack arm's 4.5791: **60 %**.
- **P-lev-d (fixed depth).** Tokens K1−K6 above 0.020 (depth 1 is off-distribution for
  this arm): **50 %**. CE at depth 6 on the paired rows within 0.03 of the unpack arm's
  4.4971: **60 %**. Movement per pass above 5 %: **35 %**.
- **P-lev-e (the panel).** At least one arm reads tokens K3−K6 above 0.002 with the CI
  above 0: **40 %**. None does (H-lev-0): **60 %**.
- **P-lev-f (cost).** Every arm within 1.15× the unpack arm's wall clock (51 min): **85 %**.

## Binding

- P-lev-b TRUE on K3−K6 ⇒ the stability terms cost the loop its depth; the next arm is the
  terms scaled down, not off, and the spike instruments decide the price.
- P-lev-c TRUE on K3−K6 ⇒ the entry is the pin; the slot loop takes the Parcae entry and
  the plain ruler's anatomy says whether the prelude entry is a fixed point of the map.
- P-lev-d TRUE ⇒ the draw is the pin; the follow-up is the draw's mean and a matched
  Poisson control, never a fixed-depth ship (E13).
- P-lev-e FALSE (no arm moves K3−K6) ⇒ H-lev-0: the reread arm (the looping slot reads the
  frozen prelude token states of its span each pass) is built and run next; the constraint,
  entry and draw lanes close for the slot loop under norm_match.
- A tripped arm goes to the divergence README with its trip step and probe; no re-run.
- NO run beyond 5,000 steps from this experiment.

## Not verified before launch

Whether `core_state_init noise` reaches `_tul_core` the way it reaches `_core_region`
(both call `self.core_init(e)`; checked on a CPU build only). The fixed-depth arm under the
hinge (E13 measured the scale mode on the M-next arm, not on the unpack arm). The
slot-state probe on an all-dim carry (it spies `prefix_project`, which is after the loop,
so it should not care). The runner's file-driven queue is new tonight.
