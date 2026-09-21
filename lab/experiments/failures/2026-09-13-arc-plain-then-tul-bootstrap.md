# Planned: plain for 5,000 steps, THEN the slot loop

Status: failure

Date: 2026-09-13 (frozen before any GPU step of the arm; no smoke of it exists at filing
time). Arc: [`2026-09-04-loop-contribution-arc.md`](2026-09-04-loop-contribution-arc.md).
One-factor partner: `slot-spandec-strict`
([`2026-09-12-arc-strict-geometry.md`](../failures/2026-09-12-arc-strict-geometry.md)).
Design note:
[`2026-09-13-plain-then-tul-bootstrap.md`](../../../.agents/notes/proposed/architecture/2026-09-13-plain-then-tul-bootstrap.md).
Config: `morph/configs/tul_slot_strict_bootstrap.yaml`.

## Wolfe's question

> "boot strap the model by training no tul for like 5k steps then use TUL for 5k steps to
> see what happens to loop contribution."

## Why it is worth a run

Thirteen slot arms have now read token K1−K6 inside [−0.0001, +0.0033], the partner at
**+0.0016 [+0.0013, +0.0019]** with K3−K6 **+0.0002**. Every one of them trained the loop,
the reader and the language model TOGETHER from step 0. So one explanation has never been
tested: at step 0 the slot state `z` is noise, the coda is learning to read tokens, and the
cheapest thing the coda can do is learn to ignore `z` — a habit that is set long before
the loop has anything to say. If that habit is what the flat curve measures, then giving
TUL a prelude and a coda that are already 5,000 steps old should move it.

This is the one lever in the arc that changes NEITHER the mechanism nor the objective: same
strict geometry, same span decoder, same 5,000 steps, same seed. Only the starting weights
differ.

## Method

`slot-strict-bootstrap` = `slot-spandec-strict`, seeded from
`/home/wolfe/morph-to/checkpoints/morph/plain-panel-norm-match/step_5000.pt` (the plain
panel ruler under `norm_match`, `step: 5000`, 266.5 M saved tensors' worth of weights,
verified present 2026-09-13).

**The schedule, and why it is `init_from` and not `resume`.** `create_lr_schedule` is a
pure function of the ABSOLUTE step, so a `training.resume` at step 5,000 with
`steps: 10000` leaves `warmup: 1000` four thousand steps in the past: the TUL phase would
open at full flat LR with NO ramp, which `lab/divergence/DIVERGENCE-README.md` §A names as
the detonating configuration and whose cure (the 1,000-step ramp) is measured at 0 of 9
draws. Nothing in the tree re-runs a ramp from a resumed step. `training.init_from` is the
trainer's documented weights-only seed: step axis back to 0, optimizer FRESH, ramp re-runs.
So the arm is **5,000 steps numbered 0…5,000** with 5,000 plain steps behind it, its
checkpoints are `step_2500.pt` / `step_5000.pt` in its own numbering, and the total training
behind its last checkpoint is 10,000 steps. It is **not compute-matched** to the partner and
every row must say so.

**The data.** `init_from` starts the token stream at its head, so
`training.data_skip_batches: 5000` fast-forwards it to where the plain run stopped. Without
it the TUL phase would take a second epoch over ~30 M tokens no other arm sees.

**What the seed cannot carry.** The seed is a PLAIN checkpoint and this arm is a
`tul.tg_restrict` model. Under `tg_restrict` MORPHAttention builds no pooled compressor and
no top-k indexer in any block; the compressed branch attends slot positions directly
instead. Measured on the real 292.5 M model against the real checkpoint (CPU, 2026-09-13):
the load has **20 missing tensors, every one TUL-owned** (`tul.*`, `tul_spandec.*`) and
**105 unexpected tensors, every one a pooled-branch tensor, 3,565,184 parameters**, which
are DROPPED. Refused: 0 and 0. So about 1.2 % of the seed's parameters are thrown away and
the compressed branch of every block restarts, while the embeddings, every CCA/window
projection, the residual carrier, every MLP and its ternary shadow, the norms, the
injections and the LM mixer all transfer. This is unavoidable — no plain checkpoint can
carry a `tg_restrict` attention — and it is a second difference from the partner, named
here rather than buried. `training.resume_plain_to_tul: true` is the opt-in that allows
exactly those two classes, enumerates them from the live model's modules, prints both lists
in full, and raises on anything else.

### Readout (the same instruments the partner was scored on)

1. `core_depth_sweep.py --depths 1,2,3,6,9,12,16 --rows 480` → token K1−K6, K3−K6,
   `spandec_ce`, paired bootstrap over rows. Partner: **+0.0016 [+0.0013, +0.0019]** /
   **+0.0002**.
2. `worth_profile.py --rows 192 --modes auto` → `zero` / `all_slots` totals and the
   per-offset bins. On a strict arm `all_slots` equals `zero` by construction; the partner
   reads **0.1865**. On math the cells HURT offset 0 by about 2 nats, so offset 0 is
   reported on its own, not folded into the total.
3. Depth-6 CE, token-paired on 480 rows against `slot-spandec-strict`
   (`span_budget_profile.py`'s pairing), plus wall clock and tok/s at step 200 (partner:
   3,337 s, 11,759 tok/s).
4. `val/slot_eff_rank` (partner **5.7598**) and `val/slot_pairwise_cos` (partner
   **0.7104**), last val of the run.
5. `slot_depth_isolation.py` — per-pass contribution, per slot.
6. The matched-compute row every slot filing carries: `plain-depth1` at 14
   block-passes/token is 0.25 nats AHEAD of spandec-mask and 0.33 ahead of the mask ruler
   at 5k.


### Method amendment, 2026-09-13 (baseline re-read; predictions unchanged)

The strict ruler's `val/slot_eff_rank` 5.7598 / `val/slot_pairwise_cos` 0.7104 quoted
above comes from the trainer's last `[VAL]` line and did NOT reproduce on the saved
checkpoint: the model's own `tul_slot_state_probe`, run by
`lab/divergence/slot_rank_anatomy.py` on the DGX Spark, reads 11.9597 / 0.5635 on the
480-row probe panel and 13.8466 / 0.5201 on the trainer's own val recipe
(`lab/experiments/results/2026-09-13-rank-anatomy/README.md`). The cause is now measured
(`lab/experiments/results/2026-09-13-rank-anatomy/discrepancy.md`, commit ab7c353): the run
logged its rank from the PRE-7a24adf probe, which rebuilt the front UNRESTRICTED on a
strict model; the checkpoint reproduces 5.7598 / 0.7104 / CE 4.4249 to four decimals under
that bare front and reads 13.85 / 0.520 under the shipped probe on the trainer's val recipe.
So the "rank 5.76" headline of this lane was an inert-instrument number; the real per-row
rank of the ruler is about 13 of ~50 slots (cosine 0.52), and every `val/slot_eff_rank`
logged before 7a24adf by a front-restricted arm (strict, or tg_restrict at scope all) is a
bare-front number. Scoring rule for every rank prediction in this file: the
partner's baseline is the SAME instrument on the SAME rows as the arm (the sweep's 480
rows, per-row median), never the trainer's logged figure. Rank predictions stated as
"vs 5.7598" are scored against that re-read baseline; the predicted DIRECTION and the
thresholds are unchanged. Also recorded from the same instrument: the prelude makes the
rank (seed 2.24 to entry 12.75 per row), six passes leave it at 13.17, and the prefix
write cuts it to 7.16 while doubling the cell count.

## Predictions (frozen)

Written before any GPU step of the arm. Probabilities are the builder's.

- **P-1 (the K-curve stays flat).** `slot-strict-bootstrap` token **K1−K6 below 0.005**.
  **75 %.** Reasoning: thirteen arms have read inside [−0.0001, +0.0033] across every lever
  tried, and this one changes neither the mechanism nor the objective. Against it: it is
  the first arm where the reader is mature when the loop starts, which is the one
  confounder the arc has never removed. Residual: 20 % in [0.005, 0.02], 5 % above 0.02.
- **P-2 (the later passes stay flat).** **K3−K6 below 0.002**, against the partner's
  +0.0002. **85 %.** Reasoning: every arm that moved K1−K6 at all moved the FIRST pass, and
  nothing here gives a second pass a referent the first does not have.
- **P-3 (the orchestrator's expectation, recorded as a hypothesis, not mine).** The mature
  coda has already learned to predict without a `z`, so it ignores the slot channel HARDER:
  `worth_profile`'s `all_slots` total is **BELOW the partner's 0.1865**. **50 %.**
  Reasoning for: the reader's habit is set by 5,000 steps of never having a `z`, and the
  gradient that would teach it to read one now competes with a converged solution. Against:
  a mature prelude hands the loop a better seed, and a better `z` is easier to read, which
  pushes `all_slots` UP. These are the two halves of the same question and the arm is worth
  running precisely because they disagree. Residual: 40 % above 0.1865, 10 % within ±0.005.
- **P-4 (the offset-0 bin).** `worth_profile`'s offset-0 bin is **not worse than the
  partner's by more than 0.05 nats**. **70 %.** Reasoning: the ~2-nat offset-0 damage is a
  math-corpus reading; on web text the bin has been mild. This prediction exists so a
  regression there cannot be waved through.
- **P-5 (CE is better, and that is NOT the result).** Depth-6 CE, token-paired on 480 rows,
  is **better than `slot-spandec-strict` by more than 0.05 nats**. **80 %.** Reasoning: the
  arm has 10,000 steps of training behind it against the partner's 5,000. A CE win here
  measures the extra 5,000 steps, not the loop. It is predicted so that nobody can later
  present it as evidence for the bootstrap. The standing rule holds: a 5k CE cannot rank
  looped against unlooped.
- **P-6 (the rank does not move).** `val/slot_eff_rank` at 5,000 is **in [4.5, 8.0]**,
  against the partner's 5.7598, and `val/slot_pairwise_cos` is **above 0.60** against
  0.7104. **70 %.** Reasoning: the row still holds one vector per span, and the register
  prereg's argument is that rank is set by that capacity, not by the weights' maturity.
  Residual: 20 % rank above 8.0 (a mature prelude spreads the seeds), 10 % below 4.5.
- **P-7 (no detonation).** The run reaches 5,000 steps with no tripwire
  (`preclip/total > 1e4` at any step ≥ 200) and no rate stop. **85 %.** Reasoning: the
  1,000-step ramp re-runs from this arm's step 0 and the slot-loop gain constraint is on,
  as on the partner. Against it: the loop's first pass now runs on a mature, higher-norm
  prelude state, which is a scale mode the ramp was not measured against, and the
  compressed branch of every block is freshly initialised under otherwise trained weights.
  `loop/core_gain_t0` leads the tripwire by ~500 steps and is the instrument to watch.
- **P-8 (the opening is a bump, not a cliff).** The first logged `[VAL]` CE (step 250) is
  **below the partner's step-250 value**. **65 %.** Reasoning: the backbone is trained; only
  the compressed branch, the TUL parameters and the optimizer are cold. Against it: dropping
  3.6 M parameters of a trained attention and re-ramping the LR is a real shock, and the
  strict geometry cuts channels the seed learned to use.

## Binding

If **P-1 fails** — the K-curve moves past 0.005 — then reader maturity was a real
confounder for thirteen arms, the whole panel has been measuring a cold coda, and every
later arm gets a bootstrap variant before it is scored.

If **P-1 holds and P-3 holds** (worth DOWN) — the mature coda ignores `z` harder and the
passes still read nothing — the "the coda never learned to read a z" explanation is dead in
the direction that matters, and the lane left is the target, not the schedule.

If **P-1 holds and P-3 fails** (worth UP) — the slot channel carries MORE while the passes
still read nothing — then what a mature model improves is the SEED, not the loop, which is
the same separation `slot_z_optimize` already reports and an argument for spending the next
arm on what `z` is graded on.

If **P-7 fails** — it detonates — the reading is about the ramp under a mature entry state,
not about the bootstrap, and the arm is re-run with the seed's own LR level before anything
else is concluded.

## Not verified before launch

* **No GPU step of this arm.** Not even a smoke. The 5090 is on the math panel.
* **The 105 dropped pooled-branch tensors have never been dropped on a RUNNING model.** The
  load is verified end to end on CPU at the real 292.5 M shape (20 missing / 105 unexpected
  / 0 refused, `load_weights_only` returned), but no forward, no backward and no step has
  been taken from those weights.
* **`training.data_skip_batches` has never fast-forwarded a real stream.** The code path it
  reuses is the resume replay, which is exercised, but the new knob is not. Its ~10-minute
  cost is the resume path's measured figure, not this arm's.
* **The 12-step smoke will pay the same ~10-minute fast-forward and load the 2.2 GB seed.**
  The runner's `smoke()` passes no overrides, so `data_skip_batches` cannot be turned off
  for it. Its wall clock is expected to be about ten minutes longer than the other arms'.
* **`plain-panel-norm-match` does NOT exist at 10,000 steps** (only `step_5000.pt` is on
  disk, checked 2026-09-13), so there is no plain 10k row to put beside this arm's CE. The
  10k-vs-5k gap in P-5 therefore has no plain control at all.
* **The optimizer state of the plain phase is discarded.** `init_from` leaves the optimizer
  fresh. Keeping it would need a `resume`, which the ramp rules out. The AdEMAMix slow EMA
  therefore restarts, and `ademamix_t_beta3: 3500` is measured against this arm's own axis.
* **Nothing here isolates the dropped compressed branch from the bootstrap.** A control that
  seeds a NON-`tg_restrict` slot arm from the same checkpoint would separate them; it is not
  queued.

## Results

Run: 5,000 TUL steps from the `plain-panel-norm-match` step-5000 seed, on the 5090 through
`run_recon.sh` at `3cb6114`, started 20:43 local 2026-09-20, DONE 21:39, tripwire HEALTHY
(`preclip/total` max 26.6 at step 3290), RATE OK 11,486 tok/s at step 200, final val_loss
4.2281. The seed's plain phase is the panel run of 2026-09-20 (16:11 to 17:0x local).
Artifacts in `../results/2026-09-13-register/`: `sweep_slot-strict-bootstrap_{2500,5000}.json`,
`worth_slot-strict-bootstrap_5000.json`, `slot_state_slot-strict-bootstrap_5000.json`,
`run_slot-strict-bootstrap.txt`, `paired_bootstrap_5000.{txt,json}`. Scored 2026-09-20,
seven days after the prereg, the arm having waited in the queue behind the fan arc.

**Depth sweep (480 rows) at 5,000.** d1 4.1329, d2 4.1315, d3 4.1312, d6 4.1310, d9 4.1313,
d12 4.1318, d16 4.1327; tokens K1−K6 **+0.0019 [+0.0016, +0.0022]**, K3−K6 **+0.0002
[+0.0000, +0.0003]**; the span decoder's own CE K1−K6 +0.0062, K3−K6 +0.0007. At 2,500:
d1 4.3031.

**Worth profile (`worth_profile.py`, 192 rows, token-weighted total / offset-0 bin).**
`all_slots` (= `zero`) **0.2287 / 0.9016** against the partner's 0.1865 / 0.7567; `shuffle`
0.1969 / 1.8509 (partner 0.1739 / 1.7391); `wrong_seed` 0.0344 / 0.4511 (partner 0.0426 /
0.6180).

**Paired (`paired_bootstrap_5000.txt`, 501,106 tokens, 490 blocks).** Depth 6: bootstrap −
`slot-spandec-strict` **−0.2164 [−0.2223, −0.2102]**; depth 1: −0.2145.

**Slot state at 5,000.** `val/slot_eff_rank` 12.44, `val/slot_pairwise_cos` 0.590 (at 4750:
12.64 / 0.579). The partner's 5.7598 / 0.7104 in the Predictions were BARE-FRONT probe
numbers from before `7a24adf` (`slot-rank-anatomy-2026-09-13`); the corrected ruler reads
13.85 / 0.52. Both are recorded; the clause is scored on the bar as written.

**Opening.** First logged `val/loss` at step 250: **4.7389** against the partner's 7.0412;
step 500: 4.7275 against 6.4147. `loop/core_gain_t0` max 10.2 at step 5, last 1.66.

**Scoring.**

- **P-1: HOLDS.** K1−K6 +0.0019 < 0.005. A coda mature for 5,000 steps before the loop
  starts reads the same nothing from passes 2 to 6 as a cold one. Reader maturity was
  not the confounder.
- **P-2: HOLDS.** K3−K6 +0.0002 < 0.002.
- **P-3: FAILS (worth UP).** `all_slots` 0.2287 > 0.1865, by 0.042. The mature coda does
  not ignore `z` harder; the slot channel carries MORE.
- **P-4: FAILS on the literal bar.** The offset-0 bin is 0.9016 against 0.7567, 0.145 past
  the 0.05 allowance. The clause was written to catch a regression of the bin; the bin
  moved the other way (removing `z` costs the first token MORE), which the wording also
  catches. Recorded as a fail with the direction named.
- **P-5: HOLDS, and is not the result.** −0.216 nats at depth 6, four times the 0.05 bar;
  10,000 steps against 5,000, no plain 10k control exists.
- **P-6: FAILS on the bar as written.** Rank 12.44 is outside [4.5, 8.0] and cosine 0.590
  is below 0.60. Against the corrected ruler (13.85 / 0.52) the arm sits in the same
  place; the bar was set on a number the 2026-09-13 correction retracted.
- **P-7: HOLDS.** No tripwire, no rate stop.
- **P-8: HOLDS.** 4.7389 < 7.0412 at step 250: a bump, not a cliff.

## Verdict

**Failure** (P-3, P-4 and P-6 fail; P-1, P-2, P-5, P-7, P-8 hold). The Binding's third
case applies: P-1 holds and P-3 fails with the worth UP. What a mature model improves is
the SEED, not the loop: the slot channel carries 0.042 nats more on the mature backbone
(0.145 more at the first token) while passes 2 to 6 still read +0.0002. The bootstrap
does not move the K-curve, so the thirteen-arm panel was not measuring a cold coda.

## Updated hypothesis

1. Reader maturity is closed as a confounder. Every later arm is scored without a
   bootstrap variant; the standing +0.002 yardstick for a slot-loop K1−K6 stands.
2. The mature backbone makes a better seed and a more used `z` (worth up, offset-0 up,
   `wrong_seed` cost down 0.043 → 0.034 because the coda now reads the seed's content
   rather than its identity). None of that reaches the loop. The lane is the target the
   loop is graded on (the LCTUL and fan lines), not the schedule.
3. P-6's bar was written against a retracted number. A prereg that quotes a partner's
   instrument must name the probe commit the number came from.
4. The dropped compressed branch and the bootstrap are still conflated (the un-queued
   control in the Risks). The opening bump (4.74 at step 250 against the partner's 7.04)
   says the drop cost little; a control that seeds a non-restricted slot arm would say
   whether it cost anything at all.
