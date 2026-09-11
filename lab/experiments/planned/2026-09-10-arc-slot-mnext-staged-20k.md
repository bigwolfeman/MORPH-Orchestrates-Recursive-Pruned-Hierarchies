# Planned: the staged slot loop at 20,000 steps, against the plain model at 20,000 steps

Status: planned

Date: 2026-09-10 (frozen before launch). Arc: `2026-09-04-loop-contribution-arc.md`.
The horizon read of
[`slot-mnext-staged`](../successes/2026-09-10-arc-slot-mnext-staged.md), whose own filing
names the open number: "The 0.08 CE lead over the ruler at 5k is the one number here that
would change the campaign if it held at 20k; it is unread past 5k." The matched-compute
control already exists:
[`norm-match-20k`](../failures/2026-09-09-arc-norm-match-recipe-reads.md). This is arc item
E5 ("20k matched wall clock on any THINK arm"), run on the first arm that produced a
forecast K-curve and a per-pass gradient separation.

## Question

Two questions, and the second is the one that decides the arc.

1. **Does the staged arm's lead hold?** At 5,000 steps `slot-mnext-staged` read 480-row CE
   4.2483 against the ruler's 4.3290. That is a lead over another SLOT arm. Against the
   PLAIN model under the same ternary rule it is not a lead at all: `norm-match-20k` read
   4.0390 on 480 rows at its own step 5,000, and TOKEN-PAIRED on the 491,520 positions the
   two sweeps share the staged arm is **+0.2099 [+0.2066, +0.2132]** nats BEHIND plain at
   matched steps and at 1.02x the wall clock. The plain model then went to 3.4516 by
   20,000. For the staged arm to lead plain at 20,000 it must close 0.210 nats while the
   target itself improves 0.587 nats. The honest question is not whether it wins; it is how
   the gap MOVES, because a gap that shrinks with training is the argument for a longer
   horizon and a gap that grows is the argument for closing the lane.
2. **Does the loop's share grow with training, as the plain loop's did?** Under `norm_match`
   the plain model's own loop contribution grew 0.1363 at 5k to 0.1562 at 10k to 0.1665 at
   15k to 0.1702 at 20k, and its K3-K6 grew 0.0107 to 0.0156. That growth is the reading
   that made the rule worth shipping. The staged slot loop's forecast curve at 5,000 is
   +0.0670 with a token curve of +0.0020. Nothing on this tree has ever measured a slot
   loop's contribution past 5,000 steps. Deep models converge slower, and every slot-loop
   verdict in this campaign is a 5,000-step verdict.

## Hypothesis

H-20k-1 (the arm's case). The staged loss gives passes 4 to 6 a job the exit needs, and at
5,000 steps the model has only just learned to do it: the own-span state at pass 3 reads CE
3.357 and the next-span CE falls 8.635 at pass 3 to 6.770 at pass 6, which is a 1.87 nat
conversion done by three passes that were previously redundant. Training a conversion takes
longer than training a projection. So the forecast K-curve holds or grows, the token curve
starts to move off the floor, and the CE gap to plain shrinks the way the norm_match rule's
own 5k deficit turned into a 20k lead.

H-20k-0 (the record's case, and the favoured one). The exit is the thing the coda reads,
and at 5,000 steps the staged arm's exit forecast (6.7698) is the ruler's exit (6.7724) to
0.003 nats. One pass already reaches it. The staged arm's K-curve is depth DEPENDENCE
without exit VALUE, which is the E13 and E14 pattern, and the E13 reading said the arms
that changed the loop's share did not change its depth. On this reading the 20k arm holds
its 5k shape: forecast K1-K6 near 0.067, token curve near 0.002, exit unchanged, and the
gap to plain roughly constant or growing because the slot apparatus spends 11 % of its row
on positions that carry no token label.

H-20k-2 (the instrument's case). The 5k CE lead over the RULER is one seed, and MORPH runs
decorrelate in 11 steps at a fixed seed with a 6.5 % median spread; n=1 comparisons on this
tree are unreadable. A 20k draw does not fix n=1, but it does add three more checkpoints of
the SAME run, so the trajectory of the gap is a within-run measurement and is readable
where a single end point is not.

## Method

`tul_slot_mnext_staged_20k` = `tul_slot_mnext_staged` with two changes and only two:
`training.steps` 5000 to 20000, and `training.ckpt_every` 2500 to 5000, so the run writes
step_5000, step_10000, step_15000 and step_20000. Everything else is the staged arm:
`tul.mux_stage_own_iters: 3` on the ruler, M-next MUX at beta 1 through the tied head,
prelude entry, slot loop, `prefix_k` 2, `max_slots` 64, hinge lambda 100 at target 0.9,
`core_fixed_point_lambda` 1.0, `slot_cot_clip` 4.0, `ternary_scale_mode: norm_match`, fused
kernels, seq 1024, batch 6, seed 1, ramp 1,000, `eval_every` 250, `gen_every` 0.

**The optimiser horizons, checked and not assumed.** A null `training.ademamix_t_beta3`
falls back to `training.steps` and silently retunes the slow EMA with the run length; it has
already flipped an arm from cure to inert. It is not null in this lineage: `base.yaml` pins
3500, `tul_to_panel.yaml` pins 3500 again, and `tul_short.yaml` pins
`ademamix_t_alpha: 1600`. A Hydra compose of `tul_slot_mnext_staged`,
`tul_slot_mnext_staged_20k` and `notul_norm_match_20k` prints 3500 and 1600 for all three:
the plain 20k control took the same horizons at 20,000 steps that the 5k slot arms take at
5,000, so this arm is comparable to BOTH. Both values are restated in the arm's own config so
the pin is visible in the file that changes the run length; the composed config is unchanged
by those two lines. `ademamix_alpha_cap` stays 3.5 and `spectral_project_cap` stays 0.0, as
on every arm of the panel.

**What the plain control is and is not.** `norm-match-20k` is a matched-COMPUTE, matched-STEP
horizon control, not a one-factor partner. It differs by the seed (`training.seed` 0 against
this arm's 1), by `core_fixed_point_lambda` (0.0 against 1.0, because the plain arm is on the
Parcae-entry recipe which has no fixed-point term), by the loop entry (noise with a learned B
against the prelude entry), and by the whole slot apparatus. The one-factor partner of this
arm is its own 5,000-step self, `slot-mnext-staged`, which shares its config exactly up to
step 5,000 except for `ckpt_every`. Both comparisons are reported and neither is dressed as
the other.

**Pairing.** The slot arm's rows carry 1014.0 tokens in 1,152 positions; the plain arm's
carry 1024 in 1024. The CE comparison is TOKEN-PAIRED from the sweep's per-token files, the
way the norm_match recipe read the paid loop against plain on 491,520 positions. A raw
480-row CE difference between a slot arm and a plain arm is not a comparison and is not
reported as one. The 5,000-step pairing is already done and is the bar P-b scores against:
`numpy.intersect1d` on the two sweeps' `tok_index` gives 491,520 shared positions, on which
the staged arm reads 4.2489 and the plain model 4.0390, a paired difference of +0.2099
[+0.2066, +0.2132] over a 400-draw bootstrap. The raw 480-row gap (4.2483 against 4.0390 =
0.2093) happens to agree to 0.001 here; that agreement is a fact about these two runs, not a
licence to skip the pairing at 20,000.

Runner `arc/run_slotloop3.sh` (file-driven queue, commit pinned in `arc/slotloop3_arms.txt`),
a 12-step smoke first, the draw under the sustained tripwire
(`lab/divergence/tripwire_sustained.py`). Rate rule: tok/s below 8,086 at step 200 skips the
arm and the queue continues.

Readout: `core_depth_sweep.py` at 5,000, 10,000, 15,000 and 20,000, depths 1,2,3,6,9,12,16,
480 rows, giving the token K-curve and BOTH staged forecast columns (`mux_local_next`, the
column comparable with every earlier arm, and `mux_local_own`, the memory column);
`worth_profile.py` and `slot_state_probe.py` at 20,000 (the runner's `slot` kind does both);
`slot_gradient_probe.py` and `slot_z_optimize.py` at 20,000 by hand. The 5,000-step readout
of this run is ALSO a replication of `slot-mnext-staged` at n=2 on the same config, which is
the only replication any arm in this campaign has had. No new wandb keys.

The numbers this arm is read against, cited once. ITS OWN 5,000-STEP SELF
`slot-mnext-staged`: 480-row CE at depth 6 **4.2483**; token K1-K6 +0.0020 [+0.0017,
+0.0024], K3-K6 +0.0023; `mux_local_next` K1-K6 +0.0670 [+0.0628, +0.0714] with next at
depth 1 / 3 / 6 = 6.8368 / 8.6350 / 6.7698 and own at 6.0645 / 3.3570 / 6.3781; worth zero at
offset 0 +0.078; z-optimisation `ce_loop` 4.0480 with the loop ENTRY worth +0.0073;
cancellation 0.442; per-pass share of the core gradient 0.164 / 0.270 / 0.491 / 0.017 / 0.023
/ 0.035 and per-pass cosine to the total +0.42 / 0.00 / +0.77 / +0.07 / +0.15 / -0.22;
12,164 tok/s at step 200; wall clock 54 min 38 s; tripwire max 53.5; smoke peak 11.07 GB;
trainer `[VAL 4750]` 4.3914; runner final val_loss 4.2988; `val/slot_eff_rank` 7.5086. THE
RULER `slot-mux-norm-match` at 5,000: CE 4.3290, token K1-K6 +0.0001, `mux_local` K1-K6
+0.0067, exit forecast 6.7724, z-opt entry +0.0015, cancellation 0.520, 52 min 38 s.
THE PLAIN CONTROL `norm-match-20k` (seed 0, Parcae entry, no slots): 480-row CE at depth 6
**4.0390** at 5,000 and **3.4516** at 20,000 (depth 1 at 20,000: 3.6217; depth 3: 3.4672);
loop K1-K6 0.1363 / 0.1562 / 0.1665 / 0.1702 at 5k / 10k / 15k / 20k with K3-K6 0.0107 at 5k
and 0.0156 [+0.0150, +0.0163] at 20k; final val_loss 3.5037; wall clock 3 h 36 min; peak
10.17 GB; tripwire max 55 at step 757.

## Predictions (frozen)

- **P-a (survival AND rate).** HEALTHY to 20,000 with no sustained tripwire AND tok/s at step
  200 at or above 8,086: **70 %**. The 5,000-step twin cleared both (tripwire max 53.5,
  12,164 tok/s), but this run is four times as long and every detonation this campaign has
  recorded happened at a step this arm's twin never reached. The plain 20k control survived.
  The 30 % is dominated by steps 5,000 to 20,000, which no slot-loop arm has ever run.
- **P-b (the arm's whole point: the gap to PLAIN).** Token-paired CE at depth 6 at 20,000,
  this arm minus `norm-match-20k`, SMALLER than the +0.2099 it sits behind at 5,000:
  **55 %**. Below 0.10: **20 %**. At or below zero, so the staged slot loop matches or beats
  the plain model at 20,000 steps: **5 %**. The 55 % rests on the one precedent for a deficit
  closing with training on this tree: the norm_match rule's own 0.08 nat deficit at 5k became
  a 0.010 lead at 20k. The 5 % is low because the slot arm spends 11 % of every row on
  positions with no token label and pays a MUX term, and no slot arm has ever been within
  0.13 nats of a plain model at any horizon.
- **P-c (its own 5k reading, the one-factor comparison).** 480-row CE at depth 6 at 20,000
  BETTER than its own 5,000-step value 4.2483 by more than 0.5 nats: **80 %**. The plain
  control improved 0.587 nats over the same interval on the same data and schedule; this
  clause is close to mechanical and is here so the file records the arm's own trajectory
  rather than only its distance from a control.
- **P-d (the loop's share, question 2).** `mux_local_next` K1-K6 at 20,000 at or above its
  5,000 value of 0.0670: **45 %**. Above 0.10: **20 %**. Below 0.03, so the forecast curve
  decays: **30 %**. The plain loop's own share GREW with training under this rule
  (0.136 to 0.170), which is the case for the first clause; against it, the staged curve is
  built out of depth 1 being bad rather than depth 6 being good, and a model with more
  training usually gets LESS bad at depth 1.
- **P-e (exit value, the number that separates dependence from worth).** The exit forecast
  (`mux_local_next` at depth 6) at 20,000 more than 0.05 nats BETTER than the ruler's 5,000
  step exit of 6.7724 after accounting for the horizon, stated as: this arm's depth-6 forecast
  at 20,000 is more than 0.30 nats below its own depth-6 forecast at 5,000 (6.7698):
  **60 %**. This is the clause that asks whether the conversion the loop learned becomes
  worth something, measured on the arm's own trajectory rather than against a 5k ruler.
- **P-f (tokens).** Token K1-K6 at 20,000 above 0.010: **25 %**. Above 0.030, the plain
  prelude-entry ruler's value: **10 %**. The token curve has never left the floor on an
  unmasked slot arm at any step; the case for it moving is only that no unmasked slot arm has
  been trained this long.
- **P-g (the reader).** The coda's exit-minus-entry CE from `slot_z_optimize.py` at 20,000
  above 0.02 nats (this arm read +0.0073 at 5,000, the largest of any unmasked arm):
  **35 %**. The quantity has grown once already on this arm against every other arm's
  0.0002 to 0.0033; four times the training is the only reason to expect it to keep growing.
- **P-h (gradient separation holds).** At least one pass still reads a NEGATIVE cosine to the
  total core-weight gradient at 20,000 (pass 6 read -0.22 at 5,000): **65 %**. The separation
  is caused by the loss shape, which does not change with the horizon; the 35 % is that the
  model may simply learn to do both jobs with one aligned update.
- **P-i (replication at 5,000).** This run's own step-5,000 sweep reads token K1-K6 and
  `mux_local_next` K1-K6 within 20 % of `slot-mnext-staged`'s 0.0020 and 0.0670: **60 %**.
  Same config, same seed, but the trainer is not bit-deterministic on this recipe (bag-mean
  atomics, fused kernels), and MORPH runs decorrelate in about 11 steps at a fixed seed.
  A miss here is information about the instrument, not about the arm.
- **P-j (cost).** Wall clock within 1.2x of 4 x 54 min 38 s, so under 4 h 22 min: **75 %**.
  The plain 20k control took 3 h 36 min at 1.0 relative rate; this arm runs at 12,164 tok/s
  against the ruler's 12,429.

## Binding

- P-b's first clause TRUE (the gap to plain shrinks) ⇒ the horizon argument is live: file the
  trajectory of the gap at 5k / 10k / 15k / 20k as the arc's E5 product, and the next question
  is 40,000 steps on the arm plus its plain control, which is Wolfe's call and is NOT queued
  by this file.
- P-b's first clause FALSE (the gap is flat or grows) with P-d's decay clause TRUE ⇒ the
  staged attachment is a 5,000-step artefact: the passes' separation does not survive
  training and the lane closes. The arc's conclusion is then written as the decision rule
  already drafted for "none THINKS", with the added fact that the one arm that DID produce a
  K-curve lost it with training.
- P-b FALSE with P-d's first clause TRUE and P-e TRUE ⇒ the loop earns more depth AND a better
  exit with training while still losing on CE to a plain model of the same compute. That is
  the most interesting outcome in this file and it is not a win: it says the slot apparatus'
  overhead, not its loop, is what costs the CE, and the next arm is the overhead (the row's
  11 % of unlabelled positions), not the attachment.
- P-a FALSE (a detonation between 5,000 and 20,000) ⇒ the trip step and probe go to
  `lab/divergence/DIVERGENCE-README.md`, and the fact that every slot-loop stability claim on
  this tree was measured on 5,000-step runs is recorded there in the same change.
- P-i FALSE ⇒ the panel's n=1 readings have a spread this campaign has never quantified on the
  slot loop; report the two draws side by side and stop quoting single-arm K-curve differences
  below that spread.
- A rate stop ⇒ the arm is skipped and the queue continues.

## Not verified before launch

The arm has not run on a GPU at any horizon past 5,000. What was run: a Hydra compose check
printing `training.steps` 20000 and `training.ckpt_every` 5000 against the staged arm's 5000
and 2500, with every other key in the panel list identical, including
`ademamix_t_beta3` 3500, `ademamix_t_alpha` 1600, `ademamix_alpha_cap` 3.5,
`spectral_project_cap` 0.0, `warmup` 1000, `eval_every` 250, `seed` 1, `batch_size` 6,
`seq_len` 1024, `ternary_scale_mode` norm_match, `tul.mux_stage_own_iters` 3,
`tul.tokens_through_core` False, `L_total` 1152; and the same compose against
`notul_norm_match_20k`, which prints the same 3500 and 1600 at the same 20,000 steps.

Not verified: no CPU forward was run for this arm, because it changes no model knob. Its
build is the staged arm's build, which ran on a GPU on 2026-09-10. The trainer's behaviour
across a 20,000-step slot-loop run is unmeasured in every respect that is not a linear
extrapolation of a 5,000-step run: memory at step 15,000, the gain hinge's behaviour once the
map has had four times the training, checkpoint disk (four checkpoints instead of two, about
2.27 GB each, measured on the staged arm's own checkpoints, against 1.2 TB free), and
the eval cadence (`eval_every` 250 gives 80 evals instead of 20, with `eval_ablations` on,
which is three extra coda passes and one extra prelude pass per eval batch). The wall-clock
prediction is a multiplication, not a measurement. The comparison against `norm-match-20k`
crosses a seed boundary and a loop-entry boundary and is a horizon reading, not an ablation.
Nothing in this change adds code or a test; the staged loss has its own suite
(`tests/test_tul_stage_targets.py`).
