# Planned: the M-next slot loop under Parcae's noise entry, terms off

Status: failure

Date: 2026-09-10 (frozen before launch; Wolfe, overnight follow-on to the slot-map levers
panel). Arc: `2026-09-04-loop-contribution-arc.md`. Follows
`failures/2026-09-10-arc-slot-map-levers.md` and its noise-entry arm
(`slot-unpack-noise-entry`, on the UNPACK coda).

## Question

The levers panel found that Parcae's noise entry does not move the slot map on the unpack
coda: the injection's ctx-channel pull converges as a geometric series by pass 3 (ratio
0.45, movement 108x then 22/8/4/2/1/0.7/0.4 %), the core blocks read 0.4-21 % on slot
states against 86-110 % on token states, and the tokens stay flat (K1-K6 +0.0019, K3-K6
+0.0001, forecast K1-K6 +0.0079). That reading is on the arm whose coda reads a single
frozen slot vector z. Does the same entry move the map, or the tokens, on the M-next arm,
whose reader is a tied-head forecast loss (MUX) into the NEXT span rather than a z-read
by the coda?

## Hypothesis

H-mnext-noise-1: the entry is coda-independent — the injection's linear pull is the same
mechanism regardless of what reads the slot state afterward, so this arm reads the same
flat map and flat tokens as the unpack noise-entry arm (H-lev-0 from the levers panel
holds here too). H-mnext-noise-0 (against): the MUX loss puts gradient pressure directly
on every iteration's output (the forecast target is read at whichever depth the row was
drawn at), so a map that is FREE to move (terms off) under an entry that must build the
state from noise gives that pressure something to act on, and K3-K6 rises above the
unpack arm's ~0.

## Method

`tul_slot_mnext_noise_entry` = `tul_slot_mux_norm_match` (the M-next, MUX, norm_match
base) plus ONE change: Parcae's entry block (`injection_channels: all`,
`injection_all_decay: 0.447`, `injection_all_dt: 0.8`, `injection_B: true`,
`core_state_init: noise`, `core_state_init_std: 0.02`) and both stability terms zeroed
(`slot_gain_lambda: 0.0`, `core_fixed_point_lambda: 0.0`; the hinge is off because it is
undefined under a noise entry, not because it was measured and rejected here).
`slot_cot_clip` 4.0 stays, as shipped.

One change vs `slot-mux-norm-match` (the levers panel's plain-entry MUX ruler, prelude
entry, both terms ON at shipped values): the entry (noise vs prelude) AND the terms
(off vs on) together, since the terms are inseparable from a noise entry (see above) —
this arm reads as one factor against that ruler only in combination.
One change vs `slot-unpack-noise-entry` (same entry, same terms-off base, unpack coda):
the coda (M-next MUX vs unpack z-read) — this is the controlled comparison.

5,000 steps, seq 1024, batch 6, seed 1, ramp 1,000, `tg_scoped_kernels`. Runner
`arc/run_slotloop3.sh` (file-driven queue, commit pinned in `arc/slotloop3_arms.txt`), a
12-step smoke first, the draw under the sustained tripwire
(`lab/divergence/tripwire_sustained.py`) and the step-200 rate rule (tok/s below the paid
loop's 8,086 at step 200 skips the arm, queue continues).

Readout: `core_depth_sweep.py` at checkpoints 2,500 and 5,000, depths
1,2,3,6,9,12,16, reporting both the token K-curve and the `mux_local` forecast K-curve on
480 rows; `worth_profile.py` and `slot_state_probe.py` at 5,000 (the runner's `slot` kind
does both automatically); `slot_anatomy.py` at 5,000 by hand (12 rows, forced depth 8,
per-pass movement and core-block MLP out/in), scored against the levers panel's noise-entry
and free-arm anatomy rows.

## Predictions (frozen)

- **P-a (survival).** HEALTHY to 5,000, no sustained tripwire (`preclip/total > 1e4` at
  step >= 200): **75 %**. Reasoning: the same entry ran HEALTHY on the unpack base
  overnight once the hinge was removed (the only draw that tripped anything was under the
  hinge, and that is not this arm's config); the MUX loss adds gradient pressure but not a
  new instability mode any arm here has shown.
- **P-b (tokens above the plain ruler).** Tokens K1-K6 at 5,000 above 0.03 (the plain
  prelude-entry ruler `plain-panel-norm-match`'s value): **15 %**. Above 0.10 (the plain
  NOISE-entry ruler's value, `E19`'s `parcae-entry` on the token-only loop): **5 %**.
  Reasoning: every slot arm run under norm_match so far, on any entry, any coda, any
  target, has read token K1-K6 at or under 0.005 (the reread arm: 0.0003; the terms-off
  and noise-entry unpack arms: 0.003 and 0.002); the plain loop's 0.033/0.185 numbers come
  from a sequence where every position loops, and the slot loop's compact sequence has
  nothing new to read per pass (the panel's structural finding). A coda swap does not
  change that geometry.
- **P-c (forecast K-curve).** `mux_local` forecast K1-K6 above 0.02 at 5,000: **25 %**.
  Reasoning: the MUX target is the one place gradient pressure lands on every iteration
  directly (F2 in the arc: memory/forecast targets earn more depth than tokens ever do on
  this tree, 0.037 vs 0.012-0.014), and this arm is the first MUX-plus-noise-entry
  combination; some upward move from the levers panel's unpack forecast number (0.0079) is
  plausible even if it does not clear 0.02. K3-K6 above 0.002 specifically: **15 %**.
- **P-d (val CE near the unpack noise-entry arm).** Val CE at 5,000 within 0.05 of
  `slot-unpack-noise-entry`'s 4.5676: **40 %**. Reasoning: the MUX loss and the unpack
  loss are different objectives (forecast-next vs reconstruct-z) on different heads, so
  there is no reason for them to land at the same CE; the M-next lineage on the prelude
  entry already reads noticeably different from the unpack lineage on other panels (the
  norm-match slot-loop panel: CE@6 vs plain +0.11 to +0.26 depending on arm).
- **P-e (cost).** Wall clock within 1.2x of `slot-mux-norm-match`'s: **80 %**. Reasoning:
  every levers-panel arm landed within 1.15x of its own base (P-lev-f TRUE at 0.85), and
  this arm changes only the entry and two loss terms, none of which are per-step FLOPs
  drivers at this scale.

## Binding

- P-b TRUE (K1-K6 > 0.10, matching the plain noise-entry ruler) => the entry + MUX
  combination is the recipe the slot loop needed; next is a 20k matched-wall-clock horizon
  read on this arm, ahead of any further lever search.
- P-b or P-c TRUE at the lower bar (tokens > 0.03 or forecast K3-K6 > 0.002) but not the
  higher bar => a partial move; report the numbers plainly, no horizon run yet, and the
  next arm is whichever of entry/MUX moved the needle further (matches the panel's own
  ambiguity-not-verdict pattern).
- Both flat (H-mnext-noise-1 holds) => the coda choice is not a lever on this entry
  either; combined with the levers panel and the reread arm, EVERY forward-side lever this
  tree has tried is flat, and the next read is the gradient-share probe already running in
  parallel (the loop's failure is credit assignment inside the backward pass, not a
  forward-side knob) — no further forward-lever arm queues without that probe's result.

## Not verified before launch

Whether the MUX loss's per-iteration forecast target interacts with the injection's
noise-built state in any way not seen on the unpack coda (untested combination). The
gain-hinge instrument's behavior at `slot_gain_lambda 0.0` under a noise entry — whether
`loss/gain_est` still logs a (unused, non-gradient) reading for diagnostic purposes, or is
skipped entirely when the term is off; not checked in this build-only pass. Whether the
smoke's 12 steps are enough to catch a MUX-plus-noise-entry build defect the CPU compose
check above did not exercise (the compose check imports the runtime and builds the model
graph; it does not run a forward/backward pass). The gradient probe of the existing
checkpoints (run in parallel by another agent) is not awaited by this arm.

## Results

Filed 2026-09-10 12:00. `slot-mnext-noise-entry` (commit `8aa6cd1`, runner `run_slotloop3.sh`):
HEALTHY to 4,999, tripwire max 142 at step 245, 14,887 tok/s at step 200 (paid loop 8,086),
smoke peak 10.10 GB, wall clock 43.6 min against the ruler `slot-mux-norm-match`'s 52.6 min
(0.83x). Trainer `[VAL 4750]` 4.4155 (the ruler 4.4486; `slot-unpack-noise-entry` 4.5676);
runner final val_loss 4.3457 (ruler 4.3775).

Sweeps, 480 rows, forced depths 1/2/3/6/9/12/16:

| step | tokens K1−K6 | tokens K3−K6 | `mux_local` K1−K6 | `mux_local` K3−K6 | CE@6 (ruler) |
| --- | --- | --- | --- | --- | --- |
| 2,500 | +0.0003 [+0.0002, +0.0004] | −0.0000 | +0.0064 [+0.0052, +0.0078] | −0.0002 | 4.7205 |
| 5,000 | +0.0004 [+0.0003, +0.0006] | +0.0000 | +0.0091 [+0.0081, +0.0101] | +0.0004 [+0.0002, +0.0006] | 4.2876 (4.3290) |

Worth profile at 5,000 (offsets 0..4): zero +0.066 [+0.058, +0.075], +0.050, +0.031, +0.027,
+0.015; shuffle +0.055, +0.047, +0.035, +0.033, +0.018. State probe at 5,000: |h| 316 at
depth 1 → 351 → 388 → 407 at depth 6 → 409 at 16; relative distance from the depth-1 state
0.48 / 0.48 / 0.57 / 0.57, cos 0.90–0.93; per-pass movement 0.48, 0.25, 0.13 (to 6), 0.04
(to 16). The state MOVES under this entry (the noise-built state grows and turns), and the
token CE does not change with it. Artifacts: `lab/experiments/results/2026-09-10-slot-mnext-noise-entry/`
(sweeps, worth, state probe, run log); npz in `ignored/experiment-artifacts/2026-09-10-slot-mnext-noise-entry/`.

## Verdict

P-a TRUE (healthy, no tripwire, rate above the bar). P-b FALSE on both bars (0.0004 vs
0.03 / 0.10). P-c FALSE on both clauses (forecast K1−K6 0.0091 < 0.02; K3−K6 0.0004 < 0.002).
P-d FALSE (0.15 nats BETTER than the unpack noise-entry arm, outside ±0.05). P-e TRUE
(0.83x). H-mnext-noise-1 holds: the entry plus the MUX on the shipped coda leaves the token
K-curve flat. Binding clause 3 applies: every forward-side lever this tree has tried is
flat, and the next arms come from the backward reading (`results/2026-09-10-slot-gradient-probe/`,
`results/2026-09-10-slot-z-optimize/`): `slot-mnext-progressive` (Bansal 2022 progressive
loss) and `slot-mnext-per-pass-lora` (Relaxed Recursive Transformers), both queued on the
same runner with their own planned files.

## Updated hypothesis

The z-optimisation probe run the same morning says why the moving state changes nothing:
on every slot arm the coda's CE with the loop's output equals its CE with the loop's ENTRY
state to 0.003 nats, while a gradient-fitted z is worth 0.9–2.6 nats to the same frozen coda.
The reader has capacity; the writer puts nothing into the directions the reader uses. The
per-pass weight gradients on the shared core are near-orthogonal across passes and come from
the MUX, not the token CE. The two queued arms test the two readings: a training rule that
scores every pass from any state, and a per-pass parameterisation so the passes stop
averaging on one map. CE at 5k is a horizon reading and ranks nothing.
