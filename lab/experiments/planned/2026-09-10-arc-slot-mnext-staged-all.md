# Planned: staged targets at EVERY non-final pass — the toy study's winner, in full

Status: planned

Date: 2026-09-10 (frozen before launch). Arc: `2026-09-04-loop-contribution-arc.md`.
Follows the four credit-assignment arms —
[`slot-mnext-progressive`](../failures/2026-09-10-arc-slot-mnext-progressive.md),
[`slot-mnext-per-pass-lora`](../failures/2026-09-10-arc-slot-mnext-per-pass-lora.md),
[`slot-mnext-mux-every-pass`](../failures/2026-09-10-arc-slot-mnext-mux-every-pass.md)
and the k=3 stage arm [`slot-mnext-staged`](2026-09-10-arc-slot-mnext-staged.md), which is
on the GPU as this file is written — and the toy study that ranks the attachments,
[`lab/toy_slot_loop/WRITEUP.md`](../../toy_slot_loop/WRITEUP.md). Design note:
[`2026-09-10-credit-assignment-in-the-slot-loop.md`](../../../.agents/notes/proposed/architecture/2026-09-10-credit-assignment-in-the-slot-loop.md).

## Question

The toy study's winning cell is `staged`: the own-span target at EVERY non-final pass, the
next-span target at the exit. 5 of 5 seeds solved a chain that needs iteration, against
2 of 5 for exit-only (the ruler's attachment), 1 of 5 for the same target at every pass
(`mux_every_pass`, which ran on MORPH and read flat) and 0 of 5 for the progressive loss.
The tree could only express the NARROW version of it — `tul.mux_stage_own_iters: 3`, ONE
intermediate pass — which is the arm running now.

This arm adds the code the toy's cell actually needs (`tul.mux_stage_all`) and runs the
attachment as the toy ran it: k becomes the FIRST supervised pass, the own-span term is
applied at passes k through T-1 on the slots whose realised depth reaches each pass, the
own terms are averaged, and the next-span term still supervises the final state alone.
Does the full attachment do what the narrow one does not?

## Hypothesis

H-all-1 (for): the toy's mechanism is that the intermediate passes get a job that is
REACHABLE (the span is already in the slot's prelude state) and DIFFERENT from the exit's,
so the shared map stops being asked for one thing six times. One own term at pass 3 gives
that job to one pass; the passes before it and the two after it are still unsupervised, so
most of the trajectory is unchanged. Supervising every non-final pass is the version the
toy measured, and it is the version whose backward showed the two jobs separating.

H-all-0 (against): MORPH's geometry is the toy's PERMISSIVE one, and under that geometry
the toy's own iterative task behaved like its one-pass control whatever the attachment was.
On this reading the attachment is not the binding constraint here and the arm reads flat,
like every loss-side arm before it.

## Method

`tul_slot_mnext_staged_all` = `tul_slot_mnext_staged` in every respect except WHERE the
own term sits: `tul.mux_stage_own_iters` 3 → 1 and `tul.mux_stage_all: false → true`. Both
compose the ruler `tul_slot_mux_norm_match` (`slot-mux-norm-match`), so the arm is also one
step from the ruler on the loss and identical to it everywhere else: M-next MUX at beta 1
through the tied head, prelude entry, hinge lambda 100 at target 0.9,
`core_fixed_point_lambda` 1.0, `slot_cot_clip` 4.0, `ternary_scale_mode: norm_match`,
seq 1024, batch 6, seed 1, 5,000 steps, ramp 1,000, `tg_scoped_kernels` off (the ruler runs
fused).

The mechanism, as implemented. `_tul_core` already keeps a live-carry trajectory for the
staged target and, under `mux_every_pass`, the per-pass supervision mask beside it
(`slot_valid & (depths >= j)`, ANDed with `~prefix` when the progressive draw is on). This
knob turns that SAME collection on rather than building a second one — one trajectory, one
set of masks in the tree. `_forward_tul` then builds `T - k` own terms
(`_tul_mux_loss(db_traj[j], slot_keep=mep_keep[j-1], target="own")` for `j` in `k … T-1`),
averages them, and keeps the ruler's own final next-span term; the loss is
`0.5 * (own mean) + 0.5 * next`, so `mux_beta` means what it means on every other arm and
the reported `mux_local` / `mux_rel` / `mux_kl` still come from the final term. Nothing is
detached: a term at pass `j` backpropagates through pass `j`'s core application and into
the prelude, and the exit's term reaches every pass. TRAINING only — an eval forward falls
back to the single pass-k own term, which is what keeps the sweep's two final-state columns
(`mux_local_own_final`, `mux_local_next_final`, both read from the FINAL state) identical
in shape to the k=3 arm's.

`mux_stage_all: false` is bit-identical to the pre-change tree, verified directly rather
than by reading the diff: the tiny CPU slot-loop model at seed 3/7 with
`mux_stage_own_iters=2` gives loss `9.359314918518066` and sha256
`189911591a8f60ee2f9dee7932461bca63b2bf5433d2355270763fe0461aefc0` over all 208 gradient
tensors both at master `fe42d85` and with the knob off after the change; the knob ON gives
`9.340977668762207`. The knob is refused without `mux_stage_own_iters > 0` and beside
`mux_every_pass`; `db_loop`, `tokens_through_core` and `mux_beta <= 0` are refused by the
`mux_stage_own_iters` block it modifies, and a test holds each. Five sabotage runs fail the
suite: dropping the per-pass mask, supervising the final state on the own term too,
detaching the trajectory, summing instead of averaging, and not collecting the trajectory.

Runner `arc/run_slotloop3.sh` (file-driven queue, commit pinned in `arc/slotloop3_arms.txt`),
a 12-step smoke first, the draw under the sustained tripwire
(`lab/divergence/tripwire_sustained.py`). Rate rule: tok/s below 8,086 at step 200 skips the
arm and the queue continues (the ruler read 12,429; `mux_every_pass`, the nearest cost
analogue, read 12,063).

Readout: `core_depth_sweep.py` at checkpoints 2,500 and 5,000, depths 1,2,3,6,9,12,16 —
the token K-curve plus the stage arm's TWO forecast columns from the final state,
`mux_local_next_final` (comparable with every earlier arm's `mux_local`) and
`mux_local_own_final`; `worth_profile.py` and `slot_state_probe.py` at 5,000 (the runner's
`slot` kind does both); `slot_gradient_probe.py` and `slot_z_optimize.py` at 5,000 by hand
— the gradient probe carries the cancellation ratio and the per-pass cosine profile, the
z-optimisation probe the coda's exit-minus-entry CE. New wandb keys: `loop/mux_stage_terms`
and `loop/mux_stage_t{j}` (one per own term, through the same pre-clip probe route as
`loop/mux_pass_*`), beside the existing `mux_stage_own` / `mux_stage_own_n` /
`mux_stage_next`. The SHAPE of the `loop/mux_stage_t*` ladder over training is the arm's own
instrument and neither hypothesis is scored on it.

The numbers this arm is read against, cited once. The one-factor comparison is
`slot-mnext-staged` (k=3), whose draw had not finished when these predictions were frozen,
so every bar below is set against the RULER `slot-mux-norm-match`, which both stage arms
compose: 480-row CE at depth 6 = **4.3290** at step 5,000; token K1−K6 +0.0001
[−0.0000, +0.0002], K3−K6 −0.0000; `mux_local` (forecast) K1−K6 +0.0067 [+0.0053, +0.0081],
K3−K6 +0.0005; wall clock 52 min 38 s for 5,000 steps; combined cancellation ratio 0.520;
per-pass cotangent share 0.168 / 0.168 / 0.166 / 0.160 / 0.154 / 0.183; per-pass cosine to
total 0.33 / 0.27 / 0.37 / 0.63 / 0.75 / 0.67; z-optimisation `ce_loop` 4.1173 with the loop
ENTRY worth +0.0015.

## Predictions (frozen)

- **P-a (survival).** HEALTHY to 5,000, no sustained tripwire (`preclip/total > 1e4` at
  step >= 200): **80 %**. The shape is `mux_every_pass`'s (live carry, one term per pass),
  which ran healthy at a tripwire max of 815 against the ruler's 78.7; the 20 % is for a
  term at pass 1 pushing directly on the first-pass scale mode, which `loop/core_gain_t0`
  leads by about 500 steps.
- **P-b (tokens).** Token K1−K6 at 5,000 above 0.03 (the plain prelude-entry ruler
  `plain-panel-norm-match`): **13 %**. Above 0.10 (the plain NOISE-entry value from E19's
  `parcae-entry`): **4 %**. The coda still reads only the final state through one door and
  the reader's indifference has survived six arms; this changes what the exit state IS, not
  who reads it.
- **P-c (forecast K-curve, the bar).** `mux_local_next` K1−K6 above 0.02 at 5,000 (the
  ruler: 0.0067): **32 %**. K3−K6 above 0.002 (the ruler: +0.0005): **22 %**. This is the
  toy's actual claim and this is the toy's actual attachment, which is why it sits above the
  k=3 arm's 28 %/18 %; it stays well under even money because the toy's Q4 finding says
  MORPH's permissive geometry may leave the loop nothing to stage a memory target FOR.
- **P-d (the reader).** The coda's exit-minus-entry CE from `slot_z_optimize.py` above 0.02
  nats (the ruler: +0.0015; every arm read so far: 0.0002–0.006): **20 %**. A 13x move on a
  number that has never left a narrow band across six very different loops.
- **P-e (a negative pass, the toy's separation signature).** At least one pass reads a
  NEGATIVE cosine to the total core-weight gradient in `slot_gradient_probe.py` (the toy's
  `staged` cell: −0.51 at the last pass; the ruler and every MORPH arm so far are positive
  at every pass): **30 %**. The sharpest falsifiable prediction here and the one tied
  directly to the toy's mechanism; set above the k=3 arm's 25 % because every non-final pass
  now carries the memory job, which is what made the toy's last pass separate.
- **P-f (cancellation).** Combined cancellation ratio BELOW the ruler's 0.520 (the toy:
  −0.604 correlation between cancellation and earning across its grid; `mux_every_pass` ROSE
  to 0.771 and read flat): **35 %**. A fall is the toy's signature of a loop doing more than
  one thing; the two MORPH arms that attacked this quantity from the training side moved it
  down slightly or up a lot.
- **P-g (val CE).** 480-row CE at depth 6 within 0.05 of the ruler's 4.3290: **50 %**.
  Recorded because every arm files it, and NOT treated as a verdict either way (Wolfe
  2026-09-09: short-horizon CE cannot rank looped against unlooped).
- **P-h (cost).** Wall clock within 1.2x of the ruler's 52 min 38 s: **85 %**. Up to `T − 1`
  extra `[B, S, V]` fp32 readouts and their backward is the same arithmetic
  `mux_every_pass` ran at 0.99x; the risk is memory (each retained logit tensor is ~75 MB at
  B=6, S=64, V=49169, and that arm's smoke peak rose from 10.1 to 12.62 GB), not FLOPs.

## Binding

- P-c TRUE and P-e TRUE ⇒ the toy's attachment transfers and the narrow k=3 form was the
  reason the first stage arm did not show it. Next: the same attachment under `tg_restrict`
  (`tul_slot_mnext_staged_mask.yaml`, queued), which is where the toy's Q4 says staging
  should show its largest effect, and a 20k horizon.
- P-c TRUE, P-e FALSE ⇒ the forecast curve moved without the separation signature; read the
  `loop/mux_stage_t*` ladder and the worth profile before deciding whether the mechanism is
  the staging or simply more supervision.
- P-c FALSE, P-f TRUE ⇒ the passes stopped agreeing without earning depth. The toy's
  within-cell reading (escaped seeds read LOWER cancellation than stuck seeds, but not every
  low-cancellation seed escapes) makes that "necessary but not sufficient", and the mask arm
  is then the one that matters, not another loss attachment.
- All flat (P-c, P-e, P-f FALSE) ⇒ loss attachment is closed on this loop. Five arms, two
  of them the toy's best and worst cells, all inside the ruler's confidence intervals. The
  honest next step is the geometry the toy's Q4 names, which is already queued as the mask
  arm, and after that the state's rank rather than the loop's training.

## Not verified before launch

The arm has not run on a GPU: everything above is a CPU build, a CPU test suite, five CPU
sabotage runs and a config compose check. `torch.compile` behaviour of a variable-length
list of per-pass loss terms at the REAL shape (B=6, S=64, L_total=1152, up to 8 passes) is
untested until the runner's 12-step smoke — the list is a comprehension over a range fixed
by the batch's realised max depth, which VARIES per step, so a recompile per distinct depth
is possible and unmeasured. Memory at the real shape is arithmetic, not a measurement. The
own term's interaction with the gain hinge (which samples ONE random grad iteration per
step, while the own term now touches every pass) is unmeasured. Uniform weights over the own
terms are a choice, not a tuned value. `mux_stage_own_n` reports the count from the FIRST
own term only, and `mux_stage_own` is the mean of the terms, so neither is comparable
one-to-one with the k=3 arm's single-term values. The one-factor partner
(`slot-mnext-staged`) had not finished when this file was frozen, so every bar here is set
against the ruler and the pairwise reading has to wait for that arm's numbers.
