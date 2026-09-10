# Planned: the slot loop (the real TUL) under the norm-match rule

Status: failure
Date: 2026-09-09 (frozen before launch; Wolfe: "our major priority is slot loop norm match
and slot loop + MUX norm match", after the correction that the paid loop
`tokens_through_core: true` is not TUL: "that is a hallucinated TUL you made in a previous
session"). Arc: `2026-09-04-loop-contribution-arc.md`. Follows the per-pass-strength panel
(`successes/2026-09-09-arc-per-pass-strength.md`) and the rule it shipped
(`base.yaml` `ternary_scale_mode: norm_match`, commit `2f3a128`). The paid-loop pair of
`2026-09-09-arc-norm-match-recipe-reads.md` is a side result, amended there.

## Question

The slot loop is TUL: one slot per span, only the slot positions loop in `_tul_core` at a
per-slot Poisson depth, `W_prefix` writes the looped state into the `prefix_k` positions,
the tokens pass the prelude and the coda once and do not loop (think once per span, decode
cheaply). Every slot-loop reading in the record (F1–F6 of the arc file: the loop finishes by
iteration 3, tokens read ≤ 0.0006 nats of slot depth, forecast K3−K6 0.000) was taken under
the absmean ternary rule, i.e. with a per-pass map whose MLP branch out/in was 0.2 on the
plain core. Under norm_match the plain core's branch reads 0.86–1.10 and its loop's K1−K6
went 0.033 → 0.185. Does the slot loop, with and without the MUX, earn depth under
norm_match, and do the tokens read it?

## Hypothesis

H-slot-1: the absmean rule was the limiter for the slot loop as well as for the plain loop:
under norm_match the slot map moves its state per pass, the MUX arm's forecast K-curve
grows past its absmean value (0.0135) and its K3−K6 leaves zero. H-slot-2: the tokens
still ignore the slot (F3 is a reader problem, not a map problem): token K1−K6 stays below
0.01 on both slot arms while the plain model on the same recipe reads > 0.12. H-slot-1′:
the gain constraint (`slot_gain_lambda` 100 at target 0.9) binds against the stronger map
and holds the slot loop at the absmean reading; the norm_match arms then show the hinge
penalty active all run with no gain in the K-curve.

## Method

Five arms, all on the think-once panel recipe (`tul_to_panel`: seq 1024, batch 6, seed 1,
5,000 steps, ramp 1000 then flat 1e-4, retention off, cap 0, alpha_cap 3.5, t_beta3 3500,
ternary backbone, `use_kernels` true, prelude state init, ctx carry, the fixed-point term
1.0 and the slot-loop gain constraint from `base.yaml`). One factor per pair. Runner
`arc/run_slotloop.sh`, commit pinned in `arc/SLOTLOOP_COMMIT`; per arm a 12-step smoke, the
draw under the sustained tripwire (`tripwire_sustained.py`), then the readouts. Order as
listed. The two recipe-reads arms (the absmean horizon resume, `norm-match-20k`) run AFTER
this panel on the same runner under their own prereg.

| arm | config | the one change | composed check |
| --- | --- | --- | --- |
| `slot-loop-norm-match` | `tul_slot_loop_norm_match` | the slot loop, no MUX (`tul_to_a1`) under `norm_match` | `tokens_through_core false`, `mux_beta` absent, `slot_seed boundary`, `emit_weight 0`, `plast_weight 1`, `prefix_k 2`, `max_slots 64`, slot depth Poisson mean 6 max 8, `bptt_depth 8` |
| `slot-mux-norm-match` | `tul_slot_mux_norm_match` | the slot loop with the MUX: M-next (`tul_to_mnext_y2`: `mux_target next`, `mux_beta 1.0`, `mux_rho 0.9`, `mux_tau 1.0`, head not detached, `slot_cot_clip 4.0`, `loop_cot_probe`), `ckpt_every 2500`, under `norm_match` | as above plus the MUX keys; `tg_restrict false` |
| `slot-loop-absmean` | `tul_slot_loop_absmean` | the first arm under `ternary_scale_mode: symmetric` (the control) | identical except the rule |
| `slot-mux-absmean` | `tul_slot_mux_absmean` | the second arm under `symmetric` (the control) | identical except the rule |
| `plain-panel-norm-match` | `notul_panel_norm_match` | the plain model (`notul` + `tul_to_panel`, `activate_at never`) under `norm_match`: the ruler on THIS recipe (the strength panel's `scale-norm-match` sits on the Parcae-entry recipe) | no layout; `tokens_through_core` is inert |

The composed diffs were printed before launch (`compose_diff.py` in the session scratch):
each slot arm differs from `plain-panel-norm-match` by `tul.activate_at`,
`tokens_through_core`, the rule where it is the control, and on the MUX arms the MUX keys
and `loop_cot_probe` only.

**Readouts.** Slot arms: `core_depth_sweep.py` at forced SLOT depths 1, 2, 3, 6, 9, 12, 16
(the sweep forces `tul.slot_mean_depth` / `slot_max_depth` on a slot-loop model; on a paid or
plain model it forces `model.cfg.mean_depth`), 480 rows, per-token files, at 2,500 and 5,000;
the token K-curve (`ce_tokens` K1−K6, K3−K6) on every arm and the forecast K-curve
(`mux_local` K1−K6, K3−K6) on the MUX arms; `worth_profile.py --rows 192` at 5,000; the
`loss/gain_est`, `loop/core_gain_t0`, `loss/fixed_point`, `preclip/total` traces from
the grad probe. Plain arm: sweep at core depths 0, 1, 2, 3, 6, 9, 12, 16; `core_anatomy.py
--rows 3 --depth 8` and `core_init_probe.py --rows 96` at 5,000. Results to
`lab/experiments/results/2026-09-09-slot-loop-norm-match/`. Loop CONTRIBUTION is the
reading; CE at 5,000 is a horizon reading and ranks nothing.

**The rate stop (Wolfe, 2026-09-09).** "If tokens per second on the slot TUL is lower than
paid TUL, stop and let me know." The runner reads the trainer's `tok/s` at step 200 of each
slot arm and compares it with the paid loop's step-200 reading on the same shape
(`tul-norm-match`: 8,086 tok/s at step 200, 6,600–8,000 later). Below that the runner kills
the arm, notes `RATE STOP`, and exits the queue. The record says the slot loop is faster
(the M-next Y2 arms on this recipe ran at 13,900–14,100 tok/s, `to-mnext-y2-g95/g98`
logs), so the stop is not expected.

## Predictions (frozen)

- **P-slot-a (survival).** HEALTHY to 5,000 by the sustained tripwire:
  `slot-loop-norm-match` **70 %**, `slot-mux-norm-match` **65 %** (a stronger map under
  the hinge), `slot-loop-absmean` **90 %**, `slot-mux-absmean` **90 %** (Y2 reached 5,000
  with zero spikes), `plain-panel-norm-match` **90 %**.
- **P-slot-b (the rule on the slot map).** `slot-mux-norm-match` forecast (`mux_local`)
  K1−K6 at 5,000 above 0.03 with the CI above 0 (absmean Y2: 0.0135): **55 %**; forecast
  K3−K6 above 0.01 (the arc's THINK bar; absmean: 0.0002): **35 %**.
- **P-slot-c (do the tokens read it).** Token K1−K6 over slot depth at 5,000 above 0.01
  with the CI above 0: `slot-loop-norm-match` **40 %**, `slot-mux-norm-match` **35 %**;
  above 0.03 on either: **20 %**. Both below 0.01 while `plain-panel-norm-match` reads
  above 0.12 (H-slot-2): **45 %**.
- **P-slot-d (the controls reproduce the record).** `slot-loop-absmean` token K1−K6 below
  0.01: **85 %**. `slot-mux-absmean` forecast K1−K6 within 0.01 of 0.0135: **70 %**.
- **P-slot-e (the hinge).** On the norm_match slot arms `loss/gain_est` (the map's typical
  gain) sits at or above 0.90 (the hinge binding) for more than half of the steps after
  1,000: **60 %**; on the absmean twins it stays in the record's 0.87–0.91 band: **80 %**.
- **P-slot-f (the ruler).** `plain-panel-norm-match` K1−K6 above 0.12 and K3−K6 above
  0.008 (the rule carries from the Parcae entry to the prelude entry): **70 %**; core MLP
  branch out/in at iteration 6 above 0.6: **75 %**.
- **P-slot-g (CE, a horizon reading).** Each slot arm's 480-row token CE at its trained
  depth against `plain-panel-norm-match`, token-paired: behind by more than 0.05:
  **65 %**; within ±0.05: **30 %**; ahead: **5 %**.
- **P-slot-h (cost).** Every slot arm above 8,086 tok/s at step 200: **90 %**; the MUX
  arms above 12,000: **75 %**; peaks under 14 GB: **85 %**.

## Binding

- P-slot-c TRUE on either norm_match slot arm (token K1−K6 > 0.01, CI above 0) ⇒ the
  slot loop earns under norm_match and the arc's F3 is dated; the next ask is that arm at
  20k (Wolfe's word) and the mask question reopens on it.
- P-slot-b TRUE and P-slot-c FALSE ⇒ H-slot-2: the map now moves but the reader does not
  use it; the next lever is the read path (`W_prefix` / `prefix_k` / where the coda reads
  the slot), not the core. Anatomy of the slot core (an instrument that does not exist
  yet) comes before any new arm.
- P-slot-b FALSE with P-slot-e TRUE (hinge binding all run) ⇒ H-slot-1′: the constraint
  and the rule fight; the next arm is the norm_match slot loop with the hinge target
  raised (0.95) or the hinge off under the sustained tripwire, ONE arm.
- P-slot-a FALSE on a norm_match arm ⇒ its trip step, `loop/core_gain_t0` and the probe go
  to the divergence README; no re-run without Wolfe.
- `RATE STOP` ⇒ the queue stops; Wolfe decides.
- NO run beyond 5,000 steps from this panel.

## Not verified before launch

`notul` composed with `tul_to_panel` (a plain model with `tul.eval_ablations true` and no
layout) builds and trains: only the 12-step smoke checks it. The sweep's slot-depth forcing
on an A1-style arm without a MUX (E13's `m12-a1` ran it at mean 12; mean 6 is the same
path). Whether the gain hinge reads its true value on the fused-kernel path
(`use_kernels true`; the eager path reads +0.07 high, `morph-eager-hinge-reads-noise`).
There is no branch out/in instrument for the slot core: `core_anatomy.py` refuses a
slot-loop model, so P-slot-f's branch reading exists for the plain ruler only.

## Results

Filed 2026-09-10 07:05. Five arms, 5,000 steps, seq 1024; the slot arms scored by
`core_depth_sweep.py` over forced slot depths (480 rows), `worth_profile.py` and
`slot_anatomy.py` (results in `results/2026-09-09-slot-loop-norm-match/`; the anatomies in
`results/2026-09-10-slot-map-levers/`). `slot-loop-absmean` ran as an orphan after the
runner's waiter self-matched its own log line (22:56); `slot-mux-absmean` was preempted at
step ~300 for the levers panel and re-ran last (05:58).

| arm | verdict | tok/s | peak | val@5k | tok K1−K6 | tok K3−K6 | fc K1−K6 | CE@6 − plain@6 (token-paired) | worth zero | gain_est mean / hinge frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| slot-loop-norm-match | HEALTHY | 12,011 | 11.7 | 4.2601 | −0.0000 [−0.0001, +0.0000] | −0.0000 | — | +0.164 [+0.156, +0.173] | 0.051 | 0.868 / 0.001 |
| slot-loop-absmean | HEALTHY | 12,843 | 11.7 | 4.2241 | +0.0000 [−0.0001, +0.0001] | −0.0000 | — | +0.109 [+0.100, +0.120] | 0.046 | 0.865 / 0.000 |
| slot-mux-norm-match | HEALTHY | 12,429 | 12.0 | 4.3775 | +0.0001 [−0.0000, +0.0002] | −0.0000 | +0.0067 [+0.0053, +0.0081] | +0.263 [+0.251, +0.275] | 0.094 | 0.887 / 0.005 |
| slot-mux-absmean | HEALTHY | 13,599 | 12.0 | 4.3457 | +0.0004 [+0.0003, +0.0005] | +0.0001 | +0.0069 [+0.0057, +0.0083] | +0.212 [+0.200, +0.226] | 0.051 | 0.883 / 0.002 |
| plain-panel-norm-match | HEALTHY | 9,525 | 10.2 | 4.0950 | +0.033 (K1−K6), K3−K6 +0.0055 | | | 0 | | |

Ruler anatomy (prelude entry, norm_match): movement 9.3/4.3/2.7/2.0/1.7/1.6/1.6 % per pass,
MLP out/in 0.03–0.19 at the last pass; the init probe entered from ZERO earns 0.47 from
depth 1 to 6 and ends 0.2 worse than the prelude entry. Slot anatomies: MLP out/in 0.5–0.7 %
(no-MUX), 6–30 % (MUX) on slot states; full-carrier movement 6/3/2/1.5/1.4/1.4/1.4 % (no-MUX)
and 65/17/9/6/4/4/3 % (MUX, the injection's geometric series).

## Verdict

- P-slot-a TRUE: all five HEALTHY.
- P-slot-b FALSE: forecast K1−K6 0.0067 (bar 0.03), K3−K6 0.0005 (bar 0.01); the rule
  does not lift the slot map's earning (absmean twin 0.0069).
- P-slot-c FALSE on both arms (0.0000 and 0.0001); the H-slot-2 clause fails too because
  the ruler does not read above 0.12.
- P-slot-d TRUE: the controls reproduce the record (0.0000; forecast 0.0069 vs 0.0135 is
  within 0.01).
- P-slot-e FALSE: `gain_est` 0.868–0.887 mean, the hinge binds under 1 % of steps on
  every arm; the constraint is inert under norm_match as under absmean.
- P-slot-f FALSE: the ruler reads K1−K6 0.033 and K3−K6 0.0055 under the prelude entry;
  the rule's 0.185 belongs to Parcae's noise entry (the entry confound, measured).
- P-slot-g TRUE (behind by more than 0.05 on every slot arm: +0.11 to +0.26 token-paired).
- P-slot-h TRUE (all above 8,086; the MUX arms above 12,000; peaks under 14 GB).

## Updated hypothesis

Under both rules the slot loop's contribution is zero on the tokens and under 0.01 on the
forecast; the rule acts on token states, not on slot states, where the core blocks output
under a third of what they output on tokens. The prelude entry flattens the plain loop too
(0.033 vs 0.185). The next panel (`2026-09-10-arc-slot-map-levers.md`, the stability
terms, the entry, the depth draw, and the reread) read flat as well: the slot loop on this
tree is a one-pass span compressor at a 0.1–0.26 nat price against the plain model at 5k,
a horizon reading. Wolfe's design correction (the coda must read z, not the prelude's
global state) is built and tested (`2026-09-09-arc-coda-reads-the-thought.md`) and does
not change the depth reading.
