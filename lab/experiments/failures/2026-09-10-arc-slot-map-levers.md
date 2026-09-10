# Planned: the slot-map levers — why the slot loop does not move its state

Status: failure
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

**Amendment 2026-09-10 01:20 (the structural arm runs tonight).** The reread was built
and tested in the hour after this file was frozen (commit `13b1cbc`, 128 CPU contract
tests, one real-size CPU step) and is queued behind the three knob arms, ahead of the
deferred controls, with its own frozen predictions in `2026-09-10-arc-slot-reread.md`.
"Not run tonight" above no longer holds; the Binding here still decides what FOLLOWS the
panel. Predictions untouched.

**Amendment 2026-09-10 03:15 (the hinge is undefined under the noise entry).** The
noise-entry arm's first draw, on the unpack base with the hinge on, read `loss/gain_est`
8.63 at step 0 (target 0.9) and a hinge penalty of 6,000 nats: the typical gain of one pass
is not a defined quantity for an entry whose first pass grows the state from 0.02-std
noise to the injected e by construction (the hinge was measured on the prelude entry only,
E14). The draw was stopped at step ~200 and the arm re-runs on the FREE base (hinge 0,
fixed point 0; `tul_slot_unpack_noise_entry` now composes `tul_slot_unpack_free`), so it is
one factor, the entry, against the free arm, which had just read the terms as not the pin
(tokens K1−K6 0.003, K3−K6 0.0001 at 5,000). P-lev-c's comparisons stay as written
against the unpack arm; the free arm's numbers are the nearer control. Predictions
untouched.

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

## Results

Filed 2026-09-10 06:10. All arms 5,000 steps, seq 1024, norm_match, scored by
`results/2026-09-10-slot-map-levers/score_levers.py` against the unpack arm
(`slot-unpack-norm-match`: tokens K1−K6 +0.0006 [+0.0004, +0.0008], K3−K6 +0.0000, CE@6
4.4971, worth zero 0.811). The plain ruler `plain-panel-norm-match` (prelude entry): tokens
K1−K6 0.018 at 2,500 and 0.033 [K3−K6 0.0055] at 5,000; entered from ZERO the same
checkpoint earns 0.47 from depth 1 to 6 and ends 0.2 worse (init probe).

| arm | survival | tok/s | val@5k | tok K1−K6 | tok K3−K6 | fc K1−K6 | CE@6 − ref | worth zero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| free (terms off) | AMBIGUOUS (one 1,070 at step 245, ramp) | 13,811 | 4.6057 | +0.0030 [+0.0025, +0.0035] | +0.0001 | +0.0044 | +0.030 | 0.864 |
| noise entry (free base) | HEALTHY | 13,986 | 4.5676 | +0.0019 [+0.0016, +0.0022] | +0.0001 | +0.0079 | −0.007 | 0.788 |
| fixed depth 6 | AMBIGUOUS (one 2,380 at step 245, ramp) | 15,306 | 4.6222 | +0.1168 [+0.1138, +0.1198] | +0.0067 [+0.0062, +0.0071] | +0.0779 | +0.052 | 0.947 |

Slot anatomy (`slot_anatomy.py`, 12 rows, 635 slots, forced depth 8, full-carrier movement
per pass and the core blocks' MLP out/in at the last pass): unpack 8.7 % then 3.8/2.2/1.6/
1.4/1.2/1.1/1.1 %, MLP 1.4–2.4 %; free 294 % then 46/25/18/14/12/10/9 % (the carrier grows
67 → 577), MLP 3–10 %; noise entry 108× then 22/8/4/2/1/0.7/0.4 % (the injection's
geometric series, ratio 0.45), MLP 5–21 %; fixed depth 9.2× then 30/12/9/4/1.5/1.5/1.8 %,
MLP 2–11 %. The plain ruler's token states under the same prelude entry: movement 9.3/
4.3/2.7/2.0/1.7/1.6/1.6 %, MLP 3–19 % at the last pass (the noise-entry plain arm of the
strength panel: 86–110 %). The noise-entry arm's first draw on the unpack base read a
6,000-nat hinge penalty at step 0 (Method amendment 03:15); the fixed-depth arm's
slot-state probe reads 0 movement at every forced depth because `slot_depth_fixed`
overrides the probe's forcing (the anatomy sets the fixed knob and is the valid reading).

## Verdict

- P-lev-a: free and fixed-depth AMBIGUOUS on one ramp reading each, no spike train;
  noise-entry HEALTHY. No detonation.
- P-lev-b FALSE: the terms are not the pin. Without them the map is expansive (the carrier
  grows 8.6× over 8 passes) and the tokens still read the same z at every depth.
- P-lev-c FALSE: the entry is not the pin. The state is built in one pass by the injection
  and converges as a geometric series; the blocks stay under 21 %.
- P-lev-d SPLIT: K1−K6 above 0.020 TRUE (0.117) and K3−K6 above 0.002 TRUE (0.0067), CE@6
  within 0.03 FALSE (+0.052). The draw makes z depth-SPECIFIC (a bowl with its minimum at
  the trained depth) and no better: depth dependence without depth value (the E6 pattern).
- P-lev-e: TRUE on the letter through the fixed-depth arm, whose curve reads
  off-distribution sensitivity (stated in Method). On the reading that matters, no arm
  earns depth. H-lev-0 holds: none of the three levers makes the slot loop earn.
- P-lev-f TRUE: every arm within 1.15× the unpack arm's wall clock.

## Updated hypothesis

The slot loop's map under every entry is the injection's linear pull on the ctx channel
(converged by pass 3) plus core blocks that output 2–20 % on slot states, against 86–110 %
on token states under the same rule and noise entry. The coda reads z (0.8 nats) and is
insensitive to its refinement; a fixed depth makes the refinement depth-specific at a
CE price. The structural arm (the reread, `2026-09-10-arc-slot-reread.md`) also read
flat: giving the pass something to read does not make the blocks compute. What is left is
not a knob on this forward: the per-pass computation on a slots-only sequence has no
gradient reason to exist while the first pass already gives the coda everything it can
use. The paid loop remains the one forward on this tree whose loop earns depth.
