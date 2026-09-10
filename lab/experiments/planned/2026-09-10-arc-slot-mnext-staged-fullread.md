# Planned: the MUX head reads the whole carrier, not its stream mean

Status: planned

Date: 2026-09-10 (frozen before launch). Arc: `2026-09-04-loop-contribution-arc.md`.
One factor against the staged arm [`slot-mnext-staged`](2026-09-10-arc-slot-mnext-staged.md),
which is on the GPU as this file is written. Source of the mechanism: finding **F2** of
[`results/2026-09-10-slot-geometry-audit/README.md`](../results/2026-09-10-slot-geometry-audit/README.md).
Design note:
[`2026-09-10-slot-loop-readout-and-attention-defects.md`](../../../.agents/notes/proposed/architecture/2026-09-10-slot-loop-readout-and-attention-defects.md).

## Question

The slot loop has two readers and they do not read the same object. `TULSlots.prefix_project`,
which writes the coda's prefix cells, projects EVERY Hyper-Connection stream separately and
hands the coda all four. The MUX head — the term that pays 7.3x what the token CE pays into
the loop state — reaches `_readout`, which collapses the four streams with an unweighted
mean before `lm_mixer` and `final_norm`. Measured on `slot-unpack-free` at step 5000: the
ENTRY state survives that mean at 0.972 of its per-stream norm and the loop's UPDATE at
0.139. The loss that trains the slot state was reading the reduction that discards the
loop's work, while the tensor the coda gets keeps it.

Does the MUX term change what the loop learns when it reads the carrier the way the coda's
reader does?

## Hypothesis

H-read-1 (for): the MUX gradient is the loop's main supervision, and it arrives through a
reduction that is blind to whatever the streams disagree about. Normalising each stream
before the average stops one stream's magnitude from deciding what the head sees, so the
MUX term can ask for structure the mean cannot represent, and the forecast K-curve moves.

H-read-0 (against, and it is specific): on THIS arm's lineage the mean is nearly symmetric
between entry and update — `slot-mux-norm-match` reads 0.581 and 0.577, not 0.972 and 0.139.
The 86 % loss is a property of `slot-unpack-free`, an arm with no stability terms whose loop
grows 7.3x. On a constrained loop the reduction may be discarding very little, in which case
this arm reads flat and F2 is a real defect with no measurable consequence here.

## Method

`tul_slot_mnext_staged_fullread` = `tul_slot_mnext_staged` (`mux_stage_own_iters: 3`, itself
the ruler `slot-mux-norm-match` plus that one knob) plus ONE change: `tul.mux_readout: full`.

The mechanism, as implemented. `"full"` applies `lm_mixer` and `final_norm` PER STREAM and
averages the `n` results; `"mean"` (the shipped path, `_readout`) averages the streams first.
Because `lm_mixer` is a per-channel-group scale plus a bias-free `nn.Linear` and the tied
head is `z @ lm_weight().t()`, `mean_n (z_n W') == (mean_n z_n) W'` exactly — so `"full"` IS
"the head applied to each stream and the logits averaged", implemented as one `[B, S, V]`
matmul rather than `n` of them, with no memory cost over the shipped path.

Why that definition and not "the head applied to the stream SUM": the sum is provably the
same arm as the mean. `final_norm` is an RMSNorm, which is scale-invariant, and everything
before it is linear, so `_readout(n · mean) == _readout(mean)` to the last bit; a test holds
it. The per-stream form is also the one consistent with `prefix_project`. The only difference
between `"mean"` and `"full"` is therefore WHERE the RMS normalisation sits, which is exactly
the quantity F2 measured.

`TULSlots.unpack` is NOT changed. Its only operation on `z` is the linear `W_bcast`, and a
linear map commutes with the stream mean, so a per-stream variant there is algebraically the
same tensor; changing what `unpack` sees needs the learned `[n·C → C]` projection the audit
names, which adds 4.2 M parameters and is a different arm. This arm does not run `bcast`.

`mux_readout: "mean"` is bit-identical to the pre-change tree, verified directly: the tiny
CPU slot-loop model at seed 3/7 gives loss `9.359314918518066` and sha256
`189911591a8f60ee2f9dee7932461bca63b2bf5433d2355270763fe0461aefc0` over all 208 gradient
tensors both at master `cffe3bb` and after the change. An unknown value is refused; a carrier
with no stream axis is refused. Three sabotage runs fail the suite (make the helper the mean
readout, make the loss ignore the knob, take the mean before the norm instead of after).

Everything else is the staged arm: M-next MUX at beta 1 through the tied head, own-span
target at pass 3, prelude entry, hinge lambda 100 at 0.9, `core_fixed_point_lambda` 1.0,
`slot_cot_clip` 4.0, `ternary_scale_mode: norm_match`, seq 1024, batch 6, seed 1, 5,000
steps, ramp 1,000.

Runner `arc/run_slotloop3.sh`, a 12-step smoke first, the draw under the sustained tripwire
(`lab/divergence/tripwire_sustained.py`). Rate rule: tok/s below 8,086 at step 200 skips the
arm (the ruler read 12,429).

Readout: `core_depth_sweep.py` at 2,500 and 5,000, depths 1,2,3,6,9,12,16 — the token
K-curve plus the stage arm's two final-state forecast columns, `mux_local_next_final` (the
one comparable with every earlier arm) and `mux_local_own_final`; `worth_profile.py` and
`slot_state_probe.py` at 5,000; `slot_gradient_probe.py` and `slot_z_optimize.py` at 5,000 by
hand, for the cancellation ratio, the per-pass cosine profile and the coda's
exit-minus-entry CE. No new wandb keys: the knob changes a readout, not a term.

The numbers this arm is read against, cited once. The one-factor partner
(`slot-mnext-staged`, k=3) had not finished when these predictions were frozen, so every bar
is set against the RULER `slot-mux-norm-match` that both compose: 480-row CE at depth 6 =
**4.3290** at 5,000; token K1−K6 +0.0001 [−0.0000, +0.0002], K3−K6 −0.0000; `mux_local`
K1−K6 +0.0067 [+0.0053, +0.0081], K3−K6 +0.0005; wall clock 52 min 38 s; cancellation ratio
0.520; per-pass cotangent 0.168 / 0.168 / 0.166 / 0.160 / 0.154 / 0.183; z-optimisation
`ce_loop` 4.1173 with the loop ENTRY worth +0.0015. The audit's stream-mean survival on that
ruler: 0.581 (entry) against 0.577 (update).

## Predictions (frozen)

- **P-a (survival).** HEALTHY to 5,000, no sustained tripwire: **85 %**. The change is one
  reduction inside an auxiliary loss; it adds no path through the loop and no parameters. The
  15 % is for the per-stream `final_norm` changing the scale the tied head sees, which
  interacts with `mux_tau` and could move the MUX gradient's magnitude early.
- **P-b (tokens).** Token K1−K6 at 5,000 above 0.03: **10 %**. Above 0.10: **3 %**. The coda's
  reader is untouched by construction — this arm changes only what the MUX term reads — so a
  token-side move has to come through a genuinely different exit state, which is a second
  step, not the first.
- **P-c (forecast K-curve, the bar).** `mux_local_next` K1−K6 above 0.02 at 5,000 (the ruler:
  0.0067): **25 %**. K3−K6 above 0.002 (the ruler: +0.0005): **15 %**. This is the curve the
  changed term supervises, so it is where the mechanism should show first; set below the two
  staged arms' bars because on this arm's lineage the mean is nearly symmetric (0.581 vs
  0.577) and there may be very little for the fix to recover.
- **P-d (the reader).** The coda's exit-minus-entry CE from `slot_z_optimize.py` above 0.02
  nats (the ruler: +0.0015): **15 %**. Lower than the staged arms' 20-22 % for the same
  reason as P-b: nothing about the coda's own path changes.
- **P-e (a negative pass).** At least one pass reads a NEGATIVE cosine to the total
  core-weight gradient in `slot_gradient_probe.py`: **20 %**. The staged term at pass 3 is
  what could produce this, and it is present on the partner arm too; this knob only changes
  how both staged terms are read, so a difference here would be a second-order effect.
- **P-f (cancellation).** Combined cancellation ratio BELOW the ruler's 0.520: **30 %**. The
  MUX side is the larger contributor (0.550 alone against the token CE's 0.601), so a change
  to what the MUX reads is the most direct route to this number of any arm in this batch.
- **P-g (val CE).** 480-row CE at depth 6 within 0.05 of the ruler's 4.3290: **60 %**. The
  MUX is an auxiliary term at beta 1 and this changes its readout, not its weight. Not a
  verdict either way.
- **P-h (cost).** Wall clock within 1.2x of the ruler's 52 min 38 s: **95 %**. The per-stream
  readout costs `lm_mixer` + `final_norm` on `[B, S, n, C]` instead of `[B, S, C]` — four
  times a 64-cell tensor — and then the SAME single `[B, S, V]` matmul. Under 0.1 % of a step.

## Binding

- P-c TRUE ⇒ the readout was load-bearing. Next: the learned `[n·C → C]` projection (the
  audit's other fix) and the same change on `prefix_project`'s side of the asymmetry, plus
  a 20k horizon.
- P-c FALSE and the stream-mean survival on this checkpoint measured near 0.5/0.5 ⇒ F2 is a
  real asymmetry with no measurable consequence on a constrained loop, and it is closed as a
  lever. Record it in the note and stop.
- P-c FALSE and the survival measured far apart (entry >> update, as on `slot-unpack-free`)
  ⇒ the reduction IS discarding the loop's work and reading it differently still does not
  help, which is a stronger negative and points back at direction, not plumbing.

## Not verified before launch

The arm has not run on a GPU: CPU build, CPU test suite, three sabotage runs and a config
compose check. The interaction of a per-stream `final_norm` with `mux_tau` and with the tied
head's scale is unmeasured beyond the tiny model, where the two readouts differ by about
3e-6 nats because the HC streams start almost equal — so the mechanism's SIZE at step 0 is
known to be tiny and its size at step 5,000 is not known at all. The stream-mean survival on
THIS arm's own checkpoint has not been measured; only the ruler's (0.581 / 0.577) and the
other two audited arms' are on record, and the binding rule above depends on it. The
one-factor partner (`slot-mnext-staged`) had not finished when this file was frozen, so the
pairwise reading has to wait. `torch.compile` behaviour of the per-stream readout is
untested until the runner's 12-step smoke.
