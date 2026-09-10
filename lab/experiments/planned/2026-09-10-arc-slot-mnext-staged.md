# Planned: staged targets on the slot loop — own span at pass 3, next span at the exit

Status: planned

Date: 2026-09-10 (frozen before launch). Arc: `2026-09-04-loop-contribution-arc.md`.
Follows the three credit-assignment arms —
[`slot-mnext-progressive`](../failures/2026-09-10-arc-slot-mnext-progressive.md),
[`slot-mnext-per-pass-lora`](../failures/2026-09-10-arc-slot-mnext-per-pass-lora.md) and
[`slot-mnext-mux-every-pass`](../failures/2026-09-10-arc-slot-mnext-mux-every-pass.md) —
and the toy study built to explain why they read flat,
[`lab/toy_slot_loop/WRITEUP.md`](../../toy_slot_loop/WRITEUP.md). Design note:
[`2026-09-10-credit-assignment-in-the-slot-loop.md`](../../../.agents/notes/proposed/architecture/2026-09-10-credit-assignment-in-the-slot-loop.md).
Prior prereg for the same knob, written 2026-09-04 and never launched:
[`2026-09-04-arc-e3-staged-targets.md`](2026-09-04-arc-e3-staged-targets.md).

## Question

The toy study fixed a task that NEEDS iteration (`compose`, S3 composition, strict
geometry: the slot loop is the ONLY cross-span path) and varied the loss attachment. Six
attachments, 5 seeds each: `staged` (own-span target at every non-final pass, next-span
target at the exit) solved the chain on 5 of 5 seeds; `exit` (the ruler's own shape, and
`deep_coda`) 2 of 5; `mux_all` (the shape of the arm that just ran on MORPH,
`slot-mnext-mux-every-pass`) 1 of 5; `progressive` 0 of 5. The mechanism the toy's backward
shows: an own-span target at the intermediate passes is reachable in about one pass, so it
does not fight the chain — it forces the state to keep the span's own contribution
decodable at every pass, which is the ingredient the final pass needs, and the per-pass
cosines to the total gradient run +0.89 down to **-0.51** at the last pass as the two jobs
separate. `slot-mnext-mux-every-pass` read flat at 5,000 steps exactly as the toy predicted
for a single-job attachment (tokens K1-K6 -0.0000, `mux_local` K1-K6 +0.0001) and its
cancellation ratio ROSE to 0.771 against the ruler's 0.520 — the toy's signature of the
failure mode, not of learning.

The tree already has the staged-target knob, `tul.mux_stage_own_iters` (`morph/model/tul.py`;
the two-term loss in `_forward_tul`, `morph/model/transformer.py` ~lines 3800-3830), built
for arc E3 on 2026-09-04 and never run on a GPU — E3's own configs predate `norm_match` and
no checkpoint directory for them exists under `/home/wolfe/morph-to/checkpoints/morph/`
(confirmed 2026-09-10). Does the own-span job at pass 3 give the intermediate passes a
reachable target that differs from the exit's, so the passes stop being redundant, the way
the toy's `staged` cell did?

## Hypothesis

H: the own-span target at pass 3 is reachable in about one pass (the slot's own span is
already in its prelude state at entry), so it does not compete with the forecast target the
way a repeated exit target does. Supervising it forces the state at pass 3 to keep the
span's own content decodable, which is exactly what a later pass needs to fold into the
forecast — so the passes after pass 3 stop reproducing pass 3's answer and start doing
something that pass 3 could not, and the shared-core update at each pass points in a more
distinct direction (a LOWER cancellation ratio, not a higher one).

## Method

`tul_slot_mnext_staged` = `tul_slot_mux_norm_match` (the ruler `slot-mux-norm-match`) plus
ONE change: `tul.mux_stage_own_iters: 3`. k=3 is the E3 prereg's original choice and the
nearest one-factor test the existing knob can express: the slot depth draw is Poisson mean
6, so most slots reach pass 3, and passes 4-6 are left for the forecast job. This is
narrower than the toy's winning cell, which supervised the own-span target at EVERY
non-final pass (1 through T-1) — expressing that in the tree needs new code (a per-pass
own-span term, the same shape as `tul.mux_every_pass` but with `target="own"`), which this
arm does not add. If k=3 reads flat, "every non-final pass" is the next question, not this
one.

5,000 steps, same recipe as the ruler in every other respect: M-next MUX at beta 1 through
the tied head, prelude entry, hinge lambda 100 at target 0.9, `core_fixed_point_lambda`
1.0, `slot_cot_clip` 4.0, `ternary_scale_mode: norm_match`, seq 1024, batch 6, seed 1, ramp
1,000, `tg_scoped_kernels`. Runner `arc/run_slotloop3.sh` (file-driven queue, commit pinned
in `arc/slotloop3_arms.txt`), a 12-step smoke first, the draw under the sustained tripwire
(`lab/divergence/tripwire_sustained.py`). Rate rule: tok/s below 8,086 at step 200 skips the
arm and the queue continues (the ruler read 12,429).

Readout: `core_depth_sweep.py` at checkpoints 2,500 and 5,000, depths 1,2,3,6,9,12,16, both
the token K-curve and the `mux_local` forecast K-curve on 480 rows. A staged arm's sweep
reports TWO forecast columns from the FINAL state at each forced depth, not one:
`mux_local_own_final` and `mux_local_next_final` (`MUX_METRICS` in `core_depth_sweep.py`,
fed by the `mux_local_own_final` / `mux_local_next_final` stats `_forward_tul` writes at
eval, `morph/model/transformer.py` ~line 3820) — both targets read from the SAME final
state the coda gets, so a stage arm's memory and forecast earning are measured on what the
reader actually sees, not on the intermediate state that trained the memory term. `next` is
the column comparable with every earlier arm's `mux_local`; `own` has no prior baseline on
this tree. `worth_profile.py` and `slot_state_probe.py` at 5,000 (the runner's `slot` kind
does both); `slot_gradient_probe.py` and `slot_z_optimize.py` at 5,000 by hand — the
gradient probe carries the cancellation ratio and the per-pass cotangent/cosine profile, and
the z-optimisation probe carries the coda's exit-minus-entry CE, the two quantities the
mechanism claims to move. New wandb keys to read: `mux_stage_own`, `mux_stage_own_n`,
`mux_stage_next` (the two training terms over time, already written by `_forward_tul`).

The ruler's numbers, cited once: 480-row CE at depth 6 = **4.3290** at step 5,000; token
K1-K6 +0.0001 [-0.0000, +0.0002], K3-K6 -0.0000; `mux_local` (forecast) K1-K6 +0.0067
[+0.0053, +0.0081], K3-K6 +0.0005; wall clock 52 min 38 s for 5,000 steps; combined
cancellation ratio 0.520; per-pass cotangent share 0.168/0.168/0.166/0.160/0.154/0.183;
per-pass cosine to total 0.33/0.27/0.37/0.63/0.75/0.67; z-optimisation `ce_loop` 4.1173 with
the loop ENTRY worth +0.0015.

## Predictions (frozen)

- **P-a (survival).** HEALTHY to 5,000, no sustained tripwire (`preclip/total > 1e4` at
  step >= 200): **80 %**. Reasoning: the staged loss adds one more term at a fixed pass with
  a live carry, the same shape `mux_every_pass` used, which ran healthy (tripwire max 815).
  The 20 % is for the own term pulling on pass 3 specifically rather than spreading across
  passes, a more concentrated gradient than the every-pass ladder's.
- **P-b (tokens).** Token K1-K6 at 5,000 above 0.03 (the plain prelude-entry ruler
  `plain-panel-norm-match`): **12 %**. Above 0.10 (the plain NOISE-entry value): **4 %**.
  Reasoning: the coda still reads only the FINAL state through one door, and every arm
  measured so far — including the ones that changed the loss entirely — has left the coda's
  exit-minus-entry CE inside a narrow band. Nothing in this change touches the reader
  directly; if it moves the token curve at all it is because the exit state itself changed,
  which is exactly what P-d asks.
- **P-c (forecast K-curve, the bar).** `mux_local_next` K1-K6 above 0.02 at 5,000 (the
  ruler: 0.0067): **28 %**. K3-K6 above 0.002 (the ruler: +0.0005): **18 %**. Reasoning:
  this is the toy's actual claim — staged targets let the LATER passes do the forecast job
  because the memory job is handled by the earlier ones. The toy's own `staged` cell solved
  its chain outright (K1-K6 +1.28, every seed). But the toy's winner supervised the own
  target at EVERY non-final pass; this arm supervises it at ONE (pass 3), which is a
  materially weaker version of the mechanism, and the toy's second finding (Q4) says MORPH's
  own permissive geometry already lets the prelude and coda compose across spans without the
  loop, which is the single biggest reason to expect this arm to read closer to flat than the
  toy's chain-solving number.
- **P-d (the reader, and the mechanism's real target).** The coda's exit-minus-entry CE from
  `slot_z_optimize.py` above 0.02 nats (the ruler: +0.0015; every arm read so far:
  0.0002-0.006): **22 %**. Reasoning: this is the number the toy's mechanism should move if
  the exit state carries something the earlier arms' exit states did not — a memory-then-
  forecast trajectory rather than one flat target repeated. It has never left a narrow band
  across five very different loops, which is why the probability is not higher, but this is
  the first arm whose mechanism (two DIFFERENT jobs on one trajectory) has an actual reason
  to move it.
- **P-e (a negative pass, the toy's separation signature).** At least one pass reads a
  NEGATIVE cosine to the total core-weight gradient in `slot_gradient_probe.py` (the toy's
  `staged` cell: pass 6 at -0.51; the ruler and every arm read so far are positive at every
  pass): **25 %**. Reasoning: this is the sharpest, most falsifiable prediction in this file
  and the one most tied to the toy's actual mechanism — two jobs in sequence pulling the
  shared weights in different directions. Set below P-c because the toy's task made the two
  jobs maximally distinct (a symbol vs. a group product); the real own-span and next-span
  M-next targets are both "predict a span's content from a tied head" and may not separate as
  cleanly.
- **P-f (cancellation, the direction that matters).** Combined cancellation ratio on the
  shared core weights BELOW the ruler's 0.520 (the toy: -0.604 correlation between
  cancellation and K1-K6 across grid A; `mux_every_pass` ROSE to 0.771 and read flat; this
  is the opposite direction from the credit-assignment note's P-d, which the toy's evidence
  overturned): **30 %**. Reasoning: a fall below 0.520 is the toy's signature of a loop doing
  more than one thing. It is set well under 50 % because the two targets here (own, next)
  are closer to each other than the toy's `compose` vs. one-pass shortcut, and because only
  ONE pass (3) carries the own term against the toy's every-non-final-pass design.
- **P-g (val CE).** 480-row CE at depth 6 within 0.05 of the ruler's 4.3290: **50 %**.
  Reasoning: not treated as a verdict either way (Wolfe 2026-09-09: short-horizon CE cannot
  rank looped against unlooped); recorded because every arm files it.
- **P-h (cost).** Wall clock within 1.2x of the ruler's 52 min 38 s: **88 %**. Reasoning: one
  extra `[B, S, V]` readout and its backward at a FIXED pass (not `T` of them, unlike
  `mux_every_pass`), so the FLOP and memory delta are a fraction of that arm's, which itself
  read 0.99x.

Honest framing before the run: the toy's geometry was STRICT by construction — the slot
loop was made the ONLY route across spans, which is not how MORPH is built. Under the toy's
own PERMISSIVE geometry (the one that matches MORPH: prelude causal over everything, coda
reading a causal chain of prefix cells), the same `compose` task behaved like the one-pass
`summary` control — one core pass reached 0.542 nats and the loop's own contribution fell
from +0.514 to +0.106 K1-K6, holding the loss attachment fixed. That is the main reason to
expect this arm to read flat despite the toy's clean staged-target result: MORPH's prelude
and coda may already be solving whatever cross-span job exists, leaving the loop nothing to
stage a memory target FOR. The mask arm (`tg_restrict`, which closes those other routes) is
the one place on the real model that has ever read a token K-curve above 0.02, and this
prereg's binding rule below reads P-c and P-e against that context rather than in isolation.

## Binding

- P-c TRUE (`mux_local_next` K1-K6 > 0.02) and P-e TRUE (a negative-cosine pass exists) ⇒
  the toy's mechanism transfers: queue the stronger form of the knob (own-span target at
  EVERY non-final pass, which needs new code) and a matched run under `tg_restrict` (the
  toy's Q4 predicts the mask arm is where staging should show the largest effect).
- P-c TRUE, P-e FALSE ⇒ the forecast curve moved without the toy's separation signature
  showing up on the gradient; read `mux_stage_own` / `mux_stage_next` over training and the
  worth profile before deciding whether to chase k or the every-non-final-pass form.
- P-c FALSE, P-f TRUE (cancellation fell) ⇒ the loop's passes stopped agreeing without
  earning depth; the toy's within-cell comparison (7 of 8 cells: escaped seeds read LOWER
  cancellation than stuck seeds, but not every low-cancellation seed escapes) says this is
  consistent with "necessary but not sufficient" — the every-non-final-pass form is still
  worth one more one-factor arm before closing.
- All flat (P-c, P-e, P-f all FALSE, token and forecast curves inside the ruler's CIs) ⇒
  this closes the fourth and most targeted credit-assignment attack on `slot-mux-norm-match`.
  Combined with the three filed arms, the honest next step is not another loss-attachment
  arm: it is the toy's Q4 finding taken at face value — test whether MORPH's own permissive
  geometry (prelude + coda attending across spans) is already doing the job any loop
  attachment could do, via the `tg_restrict` family already in the tree, read as a K-curve.

## Not verified before launch

The arm has not run on a GPU. `torch.compile` behaviour of the staged two-term loss at the
real shape (B=6, S=64, L_total=1152) is untested beyond the runner's 12-step smoke. The
`mux_stage_own_iters` code path itself has NEVER run on a GPU at all — E3 (2026-09-04) built
it, wrote CPU tests (`tests/test_tul_stage_targets.py`) and never launched; this is its first
real draw under any recipe, let alone `norm_match`. The stage arm's new sweep columns
(`mux_local_own`, `mux_local_next`) have been exercised by `core_depth_sweep.py`'s own test
suite and never against a real checkpoint. The interaction of the own term's gradient at
pass 3 with the gain hinge's random-iteration sample (the hinge samples one t per step; the
own term concentrates on iterations <= 3) is unmeasured. Memory cost at the real shape is
arithmetic, not measured; the smoke's peak is the first real number.
